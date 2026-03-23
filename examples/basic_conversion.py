#!/usr/bin/env python3
"""
Basic voice conversion example.

Converts a source audio file to sound like a target speaker.
The source content (words, timing, prosody) is preserved while
the speaker identity (timbre, pitch range) is replaced.

Usage:
    python examples/basic_conversion.py \
        --source source_speech.wav \
        --target target_speaker.wav \
        --output converted.wav \
        --device cuda:0
"""

import argparse
import time

from chatterbox_vc import VoiceConverter


def main():
    parser = argparse.ArgumentParser(description="Convert voice identity of an audio file")
    parser.add_argument("--source", required=True, help="Path to source audio (speech to convert)")
    parser.add_argument("--target", required=True, help="Path to target speaker reference audio")
    parser.add_argument("--output", default="converted.wav", help="Output path (default: converted.wav)")
    parser.add_argument("--device", default="cuda:0", help="PyTorch device (default: cuda:0)")
    args = parser.parse_args()

    # Load the model (~1.5 GB, downloaded from HuggingFace on first run)
    vc = VoiceConverter(device=args.device)

    # Convert the source audio to the target speaker's voice
    print(f"\nConverting: {args.source}")
    print(f"Target voice: {args.target}")

    t0 = time.time()
    wav = vc.convert(args.source, args.target, args.output)
    elapsed = time.time() - t0

    duration = len(wav) / vc.sample_rate
    rtf = elapsed / duration  # Real-time factor (< 1 means faster than real-time)

    print(f"\nOutput: {args.output}")
    print(f"Duration: {duration:.2f}s")
    print(f"Processing time: {elapsed:.2f}s (RTF: {rtf:.2f}x)")
    print(f"Sample rate: {vc.sample_rate} Hz")


if __name__ == "__main__":
    main()
