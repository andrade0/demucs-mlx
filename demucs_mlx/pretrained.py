# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# MLX port – Load pretrained HTDemucs models.

import inspect
import logging
import pickle
import typing as tp
import warnings
from pathlib import Path

import mlx.core as mx
import numpy as np

logger = logging.getLogger(__name__)

SOURCES = ["drums", "bass", "other", "vocals"]
ROOT_URL = "https://dl.fbaipublicfiles.com/demucs/"
REMOTE_ROOT = Path(__file__).parent.parent / 'remote'


# Meta's checkpoints are pickled with weights_only=False and reference classes
# from the original `demucs` package (e.g. demucs.htdemucs.HTDemucs). We don't
# depend on that package — we rebuild the model in MLX and only read the saved
# args/kwargs/state. Stub out any `demucs.*` class so unpickling succeeds
# without the original package installed.
class _StubUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'demucs' or module.startswith('demucs.'):
            return type(name, (), {})
        return super().find_class(module, name)


class _StubPickleModule:
    """A pickle-module shim for torch.load that tolerates missing demucs."""
    Unpickler = _StubUnpickler
    Pickler = pickle.Pickler
    load = staticmethod(pickle.load)
    dump = staticmethod(pickle.dump)


def _parse_remote_files(remote_file_list: Path) -> tp.Dict[str, str]:
    """Parse the remote files list to get model URLs."""
    root = ''
    models = {}
    for line in remote_file_list.read_text().split('\n'):
        line = line.strip()
        if line.startswith('#') or len(line) == 0:
            continue
        elif line.startswith('root:'):
            root = line.split(':', 1)[1].strip()
        else:
            sig = line.split('-', 1)[0]
            assert sig not in models
            models[sig] = ROOT_URL + root + line
    return models


def _download_model(url: str, cache_dir: Path) -> Path:
    """Download a model file if not already cached."""
    filename = url.split('/')[-1]
    cached = cache_dir / filename
    if cached.exists():
        logger.info(f"Using cached model: {cached}")
        return cached

    logger.info(f"Downloading {url}...")
    cache_dir.mkdir(parents=True, exist_ok=True)

    import urllib.request
    urllib.request.urlretrieve(url, str(cached))
    logger.info(f"Saved to {cached}")
    return cached


def _load_bag_of_models(name: str, remote_root: Path, cache_dir: Path):
    """Load a bag-of-models configuration (YAML file mapping names to sigs)."""
    import yaml

    bag_file = remote_root / (name + '.yaml')
    if not bag_file.exists():
        return None

    with open(bag_file) as f:
        bag_config = yaml.safe_load(f)

    if not isinstance(bag_config, list):
        return None

    models = _parse_remote_files(remote_root / 'files.txt')
    loaded = []
    for entry in bag_config:
        if isinstance(entry, str):
            sig = entry
            weight = None
        elif isinstance(entry, dict):
            sig = entry.get('sig') or entry.get('name', entry)
            weight = entry.get('weight')
        else:
            continue

        if sig in models:
            path = _download_model(models[sig], cache_dir)
            loaded.append((path, weight))

    return loaded if loaded else None


def load_model(name: str = 'htdemucs',
               cache_dir: tp.Optional[Path] = None) -> 'HTDemucs':
    """Load a pretrained HTDemucs model.

    Args:
        name: Model name (e.g. 'htdemucs', 'htdemucs_ft', 'htdemucs_6s').
        cache_dir: Directory to cache downloaded models.

    Returns:
        MLX HTDemucs model with loaded weights.
    """
    import torch
    from .htdemucs import HTDemucs
    from .weight_convert import convert_htdemucs_weights

    if cache_dir is None:
        cache_dir = Path.home() / '.cache' / 'demucs_mlx'

    # Try to find the model
    remote_root = REMOTE_ROOT
    models = _parse_remote_files(remote_root / 'files.txt')

    # Check if it's a bag of models
    bag_file = remote_root / (name + '.yaml')
    if bag_file.exists():
        # For bags, load just the first model
        import yaml
        with open(bag_file) as f:
            bag_config = yaml.safe_load(f)

        # Format is either: {'models': ['sig1', 'sig2']} or a list
        if isinstance(bag_config, dict) and 'models' in bag_config:
            sigs = bag_config['models']
        elif isinstance(bag_config, list):
            sigs = bag_config
        else:
            raise ValueError(f"Unexpected bag config format: {bag_config}")

        sig = sigs[0]
        if isinstance(sig, dict):
            sig = list(sig.keys())[0]

        if sig in models:
            model_path = _download_model(models[sig], cache_dir)
        else:
            raise ValueError(f"Model signature '{sig}' not found in remote files")
    elif name in models:
        model_path = _download_model(models[name], cache_dir)
    else:
        raise ValueError(
            f"Model '{name}' not found. Available: {list(models.keys())}")

    # Load PyTorch package
    logger.info(f"Loading PyTorch model from {model_path}...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        package = torch.load(model_path, map_location='cpu', weights_only=False,
                             pickle_module=_StubPickleModule)

    klass = package['klass']
    args = package['args']
    kwargs = package['kwargs']
    state = package['state']

    # Verify it's HTDemucs
    assert klass.__name__ == 'HTDemucs', \
        f"Expected HTDemucs, got {klass.__name__}"

    # Filter kwargs to match our MLX HTDemucs signature
    sig = inspect.signature(HTDemucs.__init__)
    valid_kwargs = {}
    for key, val in kwargs.items():
        if key in sig.parameters:
            valid_kwargs[key] = val
        else:
            logger.warning(f"Dropping unknown parameter: {key}")

    # Create MLX model
    logger.info("Creating MLX HTDemucs model...")
    model = HTDemucs(*args, **valid_kwargs)

    # Convert and load weights
    logger.info("Converting weights...")
    mlx_state = convert_htdemucs_weights(state)

    # Load weights into model
    _load_weights(model, mlx_state)

    logger.info(f"Model loaded: {len(model.sources)} sources, "
                f"depth={model.depth}, channels={model.channels}")
    return model


def _load_weights(model, flat_state: dict):
    """Load a flat state dict into the MLX model by walking its parameter tree.

    This handles the key mapping between PyTorch's Module tree and
    our MLX model structure.
    """
    from .weight_convert import DCONV_SEQ_MAP

    loaded = set()
    missing = []

    def _set_param(obj, attr, value):
        """Set a parameter on an MLX module."""
        if isinstance(value, np.ndarray):
            value = mx.array(value)
        setattr(obj, attr, value)

    def _load_sequential_into_dconv(dconv_layers, prefix):
        """Load PyTorch nn.Sequential DConv layers into our dict-based layers."""
        for layer_idx, layer_dict in enumerate(dconv_layers):
            layer_prefix = f"{prefix}.{layer_idx}"
            for seq_idx, our_name in DCONV_SEQ_MAP.items():
                seq_key_w = f"{layer_prefix}.{seq_idx}.weight"
                seq_key_b = f"{layer_prefix}.{seq_idx}.bias"
                seq_key_s = f"{layer_prefix}.{seq_idx}.scale"

                if our_name == 'scale':
                    if seq_key_s in flat_state:
                        _set_param(layer_dict[our_name], 'scale',
                                   flat_state[seq_key_s])
                        loaded.add(seq_key_s)
                    elif seq_key_w in flat_state:
                        # LayerScale might store as 'weight' in some versions
                        _set_param(layer_dict[our_name], 'scale',
                                   flat_state[seq_key_w])
                        loaded.add(seq_key_w)
                elif our_name.startswith('conv'):
                    mod = layer_dict[our_name]
                    if seq_key_w in flat_state:
                        _set_param(mod, 'weight', flat_state[seq_key_w])
                        loaded.add(seq_key_w)
                    if seq_key_b in flat_state:
                        _set_param(mod, 'bias', flat_state[seq_key_b])
                        loaded.add(seq_key_b)
                elif our_name.startswith('norm'):
                    mod = layer_dict[our_name]
                    if mod is not None:
                        if seq_key_w in flat_state:
                            _set_param(mod, 'weight', flat_state[seq_key_w])
                            loaded.add(seq_key_w)
                        if seq_key_b in flat_state:
                            _set_param(mod, 'bias', flat_state[seq_key_b])
                            loaded.add(seq_key_b)

    def _load_enc_dec_layer(module, prefix):
        """Load weights into an HEncLayer or HDecLayer."""
        # conv or conv_tr
        for attr in ['conv', 'conv_tr']:
            mod = getattr(module, attr, None)
            if mod is not None:
                for p in ['weight', 'bias']:
                    k = f"{prefix}.{attr}.{p}"
                    if k in flat_state:
                        _set_param(mod, p, flat_state[k])
                        loaded.add(k)

        # norm1, norm2
        for attr in ['norm1', 'norm2']:
            mod = getattr(module, attr, None)
            if mod is not None and not isinstance(mod, type(None)):
                for p in ['weight', 'bias']:
                    k = f"{prefix}.{attr}.{p}"
                    if k in flat_state:
                        _set_param(mod, p, flat_state[k])
                        loaded.add(k)

        # rewrite conv
        rewrite = getattr(module, 'rewrite_conv', None)
        if rewrite is not None:
            for p in ['weight', 'bias']:
                k = f"{prefix}.rewrite.{p}"
                if k in flat_state:
                    _set_param(rewrite, p, flat_state[k])
                    loaded.add(k)

        # DConv
        dconv = getattr(module, 'dconv_mod', None)
        if dconv is not None:
            _load_sequential_into_dconv(dconv.layers, f"{prefix}.dconv.layers")

    def _load_transformer_layer(module, prefix):
        """Load weights into a transformer encoder layer."""
        for p in ['weight', 'bias']:
            # Self-attention projections (already split by convert_htdemucs_weights)
            for proj in ['q_proj', 'k_proj', 'v_proj']:
                k = f"{prefix}.self_attn.{proj}.{p}"
                if k in flat_state:
                    _set_param(module.self_attn, proj,
                               _update_linear(getattr(module.self_attn, proj),
                                              p, flat_state[k]))
                    loaded.add(k)

            # Cross-attention projections
            for proj in ['q_proj', 'k_proj', 'v_proj']:
                k = f"{prefix}.cross_attn.{proj}.{p}"
                if k in flat_state:
                    _set_param(module.cross_attn, proj,
                               _update_linear(getattr(module.cross_attn, proj),
                                              p, flat_state[k]))
                    loaded.add(k)

            # Self-attn/cross-attn out_proj
            for attn_name in ['self_attn', 'cross_attn']:
                attn = getattr(module, attn_name, None)
                if attn is not None:
                    k = f"{prefix}.{attn_name}.out_proj.{p}"
                    if k in flat_state:
                        _set_param(attn.out_proj, p, flat_state[k])
                        loaded.add(k)

            # Linear1, Linear2 (FFN)
            for lin in ['linear1', 'linear2']:
                mod = getattr(module, lin, None)
                if mod is not None:
                    k = f"{prefix}.{lin}.{p}"
                    if k in flat_state:
                        _set_param(mod, p, flat_state[k])
                        loaded.add(k)

            # Norms
            for norm_name in ['norm1', 'norm2', 'norm3']:
                norm = getattr(module, norm_name, None)
                if norm is not None:
                    # Handle MyGroupNorm wrapping
                    actual_norm = getattr(norm, 'norm', norm)
                    k = f"{prefix}.{norm_name}.{p}"
                    if k in flat_state:
                        _set_param(actual_norm, p, flat_state[k])
                        loaded.add(k)

            # norm_out
            norm_out = getattr(module, 'norm_out_mod', None)
            if norm_out is not None:
                actual_norm = getattr(norm_out, 'norm', norm_out)
                k = f"{prefix}.norm_out.{p}"
                if k in flat_state:
                    _set_param(actual_norm, p, flat_state[k])
                    loaded.add(k)

        # LayerScale gamma_1, gamma_2
        for gamma in ['gamma_1', 'gamma_2']:
            ls = getattr(module, gamma, None)
            if ls is not None and isinstance(ls, type(model.crosstransformer.layers[0].gamma_1)):
                k = f"{prefix}.{gamma}.scale"
                if k in flat_state:
                    _set_param(ls, 'scale', flat_state[k])
                    loaded.add(k)

    def _update_linear(linear, param_name, value):
        """Update a linear layer parameter and return the linear."""
        if isinstance(value, np.ndarray):
            value = mx.array(value)
        setattr(linear, param_name, value)
        return linear

    # ── Load encoder / decoder / tencoder / tdecoder ─────────────────────
    for idx, enc in enumerate(model.encoder):
        _load_enc_dec_layer(enc, f"encoder.{idx}")

    for idx, dec in enumerate(model.decoder):
        _load_enc_dec_layer(dec, f"decoder.{idx}")

    for idx, tenc in enumerate(model.tencoder):
        _load_enc_dec_layer(tenc, f"tencoder.{idx}")

    for idx, tdec in enumerate(model.tdecoder):
        _load_enc_dec_layer(tdec, f"tdecoder.{idx}")

    # ── Load freq_emb ────────────────────────────────────────────────────
    if model.freq_emb is not None:
        k = "freq_emb.embedding.weight"
        if k in flat_state:
            _set_param(model.freq_emb.embedding, 'weight', flat_state[k])
            loaded.add(k)
        # freq_emb_scale is a float, not a parameter
        k = "freq_emb_scale"
        if k in flat_state:
            model.freq_emb_scale = float(flat_state[k])
            loaded.add(k)

    # ── Load channel up/downsamplers ─────────────────────────────────────
    for name in ['channel_upsampler', 'channel_downsampler',
                 'channel_upsampler_t', 'channel_downsampler_t']:
        mod = getattr(model, name, None)
        if mod is not None:
            for p in ['weight', 'bias']:
                k = f"{name}.{p}"
                if k in flat_state:
                    _set_param(mod, p, flat_state[k])
                    loaded.add(k)

    # ── Load CrossTransformer ────────────────────────────────────────────
    if model.crosstransformer is not None:
        ct = model.crosstransformer
        prefix = "crosstransformer"

        # norm_in, norm_in_t
        for norm_name in ['norm_in', 'norm_in_t']:
            norm = getattr(ct, norm_name, None)
            if norm is not None:
                for p in ['weight', 'bias']:
                    k = f"{prefix}.{norm_name}.{p}"
                    if k in flat_state:
                        _set_param(norm, p, flat_state[k])
                        loaded.add(k)

        # layers and layers_t
        for branch, attr in [('layers', ct.layers), ('layers_t', ct.layers_t)]:
            for idx, layer in enumerate(attr):
                layer_prefix = f"{prefix}.{branch}.{idx}"
                _load_transformer_layer(layer, layer_prefix)

    # Report
    not_loaded = set(flat_state.keys()) - loaded
    if not_loaded:
        logger.warning(f"{len(not_loaded)} weights not loaded: "
                       f"{sorted(list(not_loaded))[:20]}...")
    logger.info(f"Loaded {len(loaded)}/{len(flat_state)} weights")
