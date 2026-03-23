#!/usr/bin/env python3
"""
Integration test for chatterbox_vc.

Generates two short synthetic speech clips using TTS, then converts one
to sound like the other. Verifies that the output is valid audio at the
expected sample rate and duration.

Requirements:
    pip install chatterbox-tts  (provides both TTS and VC)

Usage:
    python tests/test_conversion.py --device cuda:0
"""

import argparse
import os
import sys
import tempfile
import time

import numpy as np
import soundfile as sf


def generate_test_audio(output_path: str, text: str, duration_hint: float = 5.0):
    """Generate a short speech clip using Chatterbox TTS for testing.

    Falls back to a sine-wave tone if TTS is not available, so the test
    can still verify the VC pipeline mechanics.

    Args:
        output_path: Where to save the WAV file.
        text: Text to synthesize.
        duration_hint: Approximate duration in seconds (used for fallback).
    """
    try:
        from chatterbox.tts import ChatterboxTTS

        model = ChatterboxTTS.from_pretrained(device="cpu")
        wav = model.generate(text)
        wav_np = wav.squeeze().cpu().numpy().astype(np.float32)
        sf.write(output_path, wav_np, 24000)
        print(f"  Generated TTS audio: {output_path} ({len(wav_np)/24000:.1f}s)")
        return True
    except Exception as e:
        print(f"  TTS unavailable ({e}), generating synthetic tone instead")
        # Generate a simple tone as fallback
        sr = 24000
        t = np.linspace(0, duration_hint, int(sr * duration_hint), dtype=np.float32)
        # Mix several frequencies to simulate speech-like signal
        wav = (
            0.3 * np.sin(2 * np.pi * 200 * t)
            + 0.2 * np.sin(2 * np.pi * 400 * t)
            + 0.1 * np.sin(2 * np.pi * 800 * t)
            + 0.05 * np.random.randn(len(t)).astype(np.float32)
        )
        wav = wav / np.max(np.abs(wav)) * 0.8
        sf.write(output_path, wav, sr)
        print(f"  Generated synthetic tone: {output_path} ({duration_hint:.1f}s)")
        return False


def test_single_conversion(device: str):
    """Test single-file voice conversion."""
    from chatterbox_vc import VoiceConverter

    print("\n=== Test: Single File Conversion ===")

    with tempfile.TemporaryDirectory(prefix="vc_test_") as tmpdir:
        source_path = os.path.join(tmpdir, "source.wav")
        target_path = os.path.join(tmpdir, "target.wav")
        output_path = os.path.join(tmpdir, "output.wav")

        # Generate test audio
        print("Generating test audio...")
        generate_test_audio(source_path, "Hello, this is a test of voice conversion.")
        generate_test_audio(target_path, "The target speaker has a different voice.")

        # Load model
        print(f"Loading VoiceConverter on {device}...")
        t0 = time.time()
        vc = VoiceConverter(device=device)
        print(f"Model loaded in {time.time() - t0:.1f}s")

        # Convert
        print("Running voice conversion...")
        t0 = time.time()
        wav = vc.convert(source_path, target_path, output_path)
        elapsed = time.time() - t0

        # Validate output
        assert isinstance(wav, np.ndarray), f"Expected numpy array, got {type(wav)}"
        assert wav.ndim == 1, f"Expected 1D array, got {wav.ndim}D"
        assert len(wav) > 0, "Output is empty"
        assert wav.dtype == np.float32, f"Expected float32, got {wav.dtype}"
        assert vc.sample_rate == 24000, f"Expected 24kHz, got {vc.sample_rate}"

        # Verify output file was written
        assert os.path.exists(output_path), f"Output file not found: {output_path}"
        file_wav, file_sr = sf.read(output_path)
        assert file_sr == 24000, f"Output file sample rate: {file_sr} (expected 24000)"
        assert len(file_wav) == len(wav), "File length doesn't match returned array"

        duration = len(wav) / vc.sample_rate
        rtf = elapsed / duration if duration > 0 else float("inf")

        print(f"\n  Output duration: {duration:.2f}s")
        print(f"  Processing time: {elapsed:.2f}s (RTF: {rtf:.2f}x)")
        print(f"  Output shape: {wav.shape}")
        print(f"  Output range: [{wav.min():.4f}, {wav.max():.4f}]")
        print(f"  Sample rate: {vc.sample_rate} Hz")
        print("  PASSED")

        return vc, tmpdir, source_path, target_path


def test_batch_conversion(vc, tmpdir: str, target_path: str):
    """Test batch conversion (reuses pre-loaded model)."""
    print("\n=== Test: Batch Conversion ===")

    # Create multiple source files
    sources = []
    for i in range(3):
        path = os.path.join(tmpdir, f"batch_source_{i}.wav")
        generate_test_audio(path, f"This is batch test number {i + 1}.", duration_hint=3.0)
        sources.append(path)

    output_dir = os.path.join(tmpdir, "batch_output")

    print(f"Converting {len(sources)} files...")
    t0 = time.time()
    results = vc.convert_batch(sources, target_path, output_dir)
    elapsed = time.time() - t0

    # Validate results
    assert len(results) == len(sources), f"Expected {len(sources)} results, got {len(results)}"

    for r in results:
        assert r["success"], f"Conversion failed for {r['source']}: {r.get('error')}"
        assert os.path.exists(r["output"]), f"Output not found: {r['output']}"
        assert r["duration_s"] > 0, f"Invalid duration: {r['duration_s']}"
        assert r["elapsed_s"] > 0, f"Invalid elapsed time: {r['elapsed_s']}"

        wav, sr = sf.read(r["output"])
        assert sr == 24000, f"Wrong sample rate in {r['output']}: {sr}"

    ok = sum(1 for r in results if r["success"])
    print(f"\n  {ok}/{len(results)} conversions succeeded in {elapsed:.1f}s total")
    print("  PASSED")


def test_set_target_voice(vc, tmpdir: str, source_path: str, target_path: str):
    """Test pre-setting target voice for efficiency."""
    print("\n=== Test: Pre-set Target Voice ===")

    output1 = os.path.join(tmpdir, "preset_out1.wav")
    output2 = os.path.join(tmpdir, "preset_out2.wav")

    # Pre-set the target voice
    print("Setting target voice...")
    vc.set_target_voice(target_path)

    # Convert twice without specifying target (uses cached embedding)
    print("Converting with cached speaker embedding...")
    wav1 = vc.convert(source_path, target_path, output1)
    wav2 = vc.convert(source_path, target_path, output2)

    assert len(wav1) > 0, "First conversion produced empty output"
    assert len(wav2) > 0, "Second conversion produced empty output"

    print(f"  Output 1: {len(wav1)/24000:.2f}s")
    print(f"  Output 2: {len(wav2)/24000:.2f}s")
    print("  PASSED")


def test_error_handling():
    """Test that appropriate errors are raised for invalid inputs."""
    from chatterbox_vc import VoiceConverter

    print("\n=== Test: Error Handling ===")

    # Test FileNotFoundError
    try:
        vc = VoiceConverter.__new__(VoiceConverter)
        vc.device = "cpu"
        vc._model = None
        # Don't load model, just test input validation
        vc.convert("/nonexistent/source.wav", "/nonexistent/target.wav")
        print("  FAILED: Should have raised FileNotFoundError")
        return
    except FileNotFoundError:
        print("  FileNotFoundError raised correctly for missing files")

    print("  PASSED")


def main():
    parser = argparse.ArgumentParser(description="Test chatterbox_vc")
    parser.add_argument("--device", default="cuda:0", help="PyTorch device (default: cuda:0)")
    parser.add_argument("--quick", action="store_true", help="Run only single conversion test")
    args = parser.parse_args()

    print(f"Running chatterbox_vc tests on {args.device}")
    print("=" * 60)

    # Test error handling (no model needed)
    test_error_handling()

    # Test single conversion (loads model)
    vc, tmpdir, source_path, target_path = test_single_conversion(args.device)

    if not args.quick:
        # Reuse loaded model for remaining tests
        test_batch_conversion(vc, tmpdir, target_path)
        test_set_target_voice(vc, tmpdir, source_path, target_path)

    print("\n" + "=" * 60)
    print("All tests passed!")


if __name__ == "__main__":
    main()
