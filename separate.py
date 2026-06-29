#!/usr/bin/env python3
"""Separate music sources with HTDemucs on Apple Silicon (MLX).

Usage:
    demucs-mlx song.mp3
    demucs-mlx song.wav --stems vocals drums
    demucs-mlx song.mp3 -o output_dir/
"""

import argparse
import os
import time


def main():
    parser = argparse.ArgumentParser(
        prog="demucs-mlx",
        description="Separate a song into stems (drums, bass, other, vocals) "
                    "using HTDemucs on Apple Silicon with MLX.",
        epilog="Examples:\n"
               "  demucs-mlx song.mp3\n"
               "  demucs-mlx song.mp3 --stems vocals\n"
               "  demucs-mlx song.mp3 --stems vocals drums -o my_stems/\n"
               "  demucs-mlx song.mp3 --shifts 3 --float32\n"
               "  demucs-mlx song.mp3 -n htdemucs_ft\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("input",
                        help="input audio file (WAV, MP3, FLAC, OGG, etc.)")

    model_group = parser.add_argument_group("model")
    model_group.add_argument("-n", "--name", default="htdemucs",
                             metavar="NAME",
                             help="model name: htdemucs, htdemucs_ft, "
                                  "htdemucs_6s (default: htdemucs)")

    output_group = parser.add_argument_group("output")
    output_group.add_argument("-o", "--output", default=None,
                              metavar="DIR",
                              help="output directory "
                                   "(default: ./separated/<model>/<song>/)")
    output_group.add_argument("--stems", nargs="+", default=None,
                              metavar="STEM",
                              help="which stems to save, choose from: "
                                   "drums bass other vocals (default: all)")
    output_group.add_argument("--mp3", action="store_true",
                              help="save stems as MP3 instead of WAV")
    output_group.add_argument("--float32", action="store_true",
                              help="save as float32 WAV instead of int16")

    quality_group = parser.add_argument_group("quality")
    quality_group.add_argument("--shifts", type=int, default=1,
                               metavar="N",
                               help="random shifts for better quality, "
                                    "slower (default: 1)")
    quality_group.add_argument("--overlap", type=float, default=0.25,
                               metavar="F",
                               help="overlap between chunks, 0.0 to 1.0 "
                                    "(default: 0.25)")
    quality_group.add_argument("--no-split", action="store_true",
                               help="process the whole track at once "
                                    "instead of chunking (uses more memory)")

    args = parser.parse_args()

    import mlx.core as mx
    import numpy as np
    import soundfile as sf
    from demucs_mlx.pretrained import load_model
    from demucs_mlx.apply import apply_model

    # Load audio
    print(f"Loading audio: {args.input}")
    wav, sr = sf.read(args.input, dtype='float32')
    if wav.ndim == 1:
        wav = wav[:, None]
    # wav: [T, C] → [1, C, T]
    wav = wav.T[None, :, :]
    print(f"Audio: {wav.shape[2] / sr:.1f}s, {wav.shape[1]}ch, {sr}Hz")

    # Load model
    print(f"Loading model: {args.name}")
    t0 = time.time()
    model = load_model(args.name)
    print(f"Model loaded in {time.time() - t0:.1f}s "
          f"({len(model.sources)} sources: {model.sources})")

    # Resample if needed
    if sr != model.samplerate:
        print(f"Resampling {sr}Hz → {model.samplerate}Hz...")
        try:
            import librosa
        except ImportError:
            raise SystemExit(
                f"Input is {sr}Hz but {args.name} expects "
                f"{model.samplerate}Hz. Install librosa to enable automatic "
                "resampling:\n\n    pip install librosa\n\n"
                "(or reinstall demucs-mlx with the 'resample' extra), or "
                "supply audio already at the model's sample rate.")
        wav_np = wav[0]  # [C, T]
        resampled = []
        for ch in range(wav_np.shape[0]):
            resampled.append(librosa.resample(
                wav_np[ch], orig_sr=sr, target_sr=model.samplerate))
        wav = np.stack(resampled)[None, :, :]
        sr = model.samplerate

    mix = mx.array(wav)

    # Separate
    print(f"Separating with {args.shifts} shift(s)...")
    t0 = time.time()
    sources = apply_model(
        model, mix, shifts=args.shifts,
        split=not args.no_split, overlap=args.overlap,
        progress=True)
    mx.eval(sources)
    sep_time = time.time() - t0
    print(f"\nSeparated in {sep_time:.1f}s "
          f"({wav.shape[2] / sr / sep_time:.2f}x realtime)")

    # Save
    sources_np = np.array(sources[0])  # [S, C, T]
    stem_names = args.stems or model.sources
    basename = os.path.splitext(os.path.basename(args.input))[0]
    out_dir = args.output or os.path.join("separated", args.name)
    out_dir = os.path.join(out_dir, basename)
    os.makedirs(out_dir, exist_ok=True)

    for i, src_name in enumerate(model.sources):
        if src_name not in stem_names:
            continue
        stem = sources_np[i].T  # [T, C]
        ext = "mp3" if args.mp3 else "wav"
        out_path = os.path.join(out_dir, f"{src_name}.{ext}")
        subtype = 'FLOAT' if args.float32 else 'PCM_16'
        sf.write(out_path, stem, sr, subtype=subtype)
        print(f"Saved: {out_path}")

    print(f"\nDone! Output in: {out_dir}")


if __name__ == "__main__":
    main()
