#!/usr/bin/env python3
"""
Batch voice conversion example.

Converts multiple source files to the same target speaker identity.
The target speaker embedding is extracted once and reused for all
source files, making this more efficient than converting one by one.

Usage:
    python examples/batch_conversion.py \
        --sources audio1.wav audio2.wav audio3.wav \
        --target target_speaker.wav \
        --output-dir ./converted/ \
        --device cuda:0
"""

import argparse

from chatterbox_vc import VoiceConverter


def main():
    parser = argparse.ArgumentParser(description="Batch voice conversion")
    parser.add_argument("--sources", nargs="+", required=True, help="Source audio files")
    parser.add_argument("--target", required=True, help="Target speaker reference audio")
    parser.add_argument("--output-dir", default="./converted", help="Output directory")
    parser.add_argument("--device", default="cuda:0", help="PyTorch device")
    args = parser.parse_args()

    vc = VoiceConverter(device=args.device)

    print(f"\nConverting {len(args.sources)} files to voice of: {args.target}")
    results = vc.convert_batch(args.sources, args.target, args.output_dir)

    print(f"\n{'Source':<40} {'Output':<40} {'Duration':<10} {'Time':<10} {'Status'}")
    print("-" * 120)
    for r in results:
        src = r["source"].split("/")[-1]
        out = r["output"].split("/")[-1]
        if r["success"]:
            print(f"{src:<40} {out:<40} {r['duration_s']:<10.2f} {r['elapsed_s']:<10.2f} OK")
        else:
            print(f"{src:<40} {out:<40} {'--':<10} {r['elapsed_s']:<10.2f} FAILED: {r['error']}")

    ok = sum(1 for r in results if r["success"])
    print(f"\n{ok}/{len(results)} conversions succeeded")


if __name__ == "__main__":
    main()
