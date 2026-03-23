"""
Core voice conversion module.

This module provides the VoiceConverter class, a high-level wrapper around
Chatterbox's S3Gen voice conversion model. It handles model loading, audio
I/O, resampling, and provides both one-shot and batch conversion APIs.

Model Architecture Overview:
    Chatterbox VC uses a three-stage pipeline:

    1. S3Tokenizer: Encodes source speech at 16kHz into discrete content tokens
       at 25Hz (one token per 40ms). These tokens capture *what* is said (phonetic
       content, prosody timing) but discard speaker identity.

    2. S3Gen (Flow-Matching Decoder): Takes the content tokens plus a speaker
       embedding (from the target voice) and generates a mel-spectrogram via
       conditional flow matching (CFM). The CFM iteratively refines noise into
       a mel-spectrogram over N timesteps (default 10).

    3. HiFi-GAN Vocoder: Converts the mel-spectrogram into a 24kHz waveform.
       Includes an F0 (pitch) predictor for natural intonation.

    Finally, a PerTH watermark is embedded (imperceptible, survives MP3 compression).
"""

import os
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import soundfile as sf
import torch

# Output sample rate of the Chatterbox S3Gen model (fixed by architecture).
OUTPUT_SAMPLE_RATE = 24000


class VoiceConverter:
    """High-level voice conversion interface built on Chatterbox S3Gen.

    Converts the speaker identity of source audio to match a target speaker
    while preserving linguistic content and prosody.

    Args:
        device: PyTorch device string. Examples: "cuda:0", "cuda", "cpu".
            GPU is strongly recommended -- CPU inference is ~20x slower.
        model_path: Optional path to a local checkpoint directory containing
            ``s3gen.safetensors`` and optionally ``conds.pt``. If None,
            weights are automatically downloaded from HuggingFace
            (``ResembleAI/chatterbox``).

    Example:
        >>> vc = VoiceConverter(device="cuda:0")
        >>> vc.convert("source.wav", "target_speaker.wav", "output.wav")

    Note:
        The first call triggers model download (~1.5 GB) which may take
        a few minutes. Subsequent calls use the HuggingFace cache.
    """

    def __init__(
        self,
        device: str = "cuda:0",
        model_path: Optional[str] = None,
    ):
        self.device = device
        self._model = None
        self._model_path = model_path
        self._load_model()

    def _load_model(self):
        """Load the ChatterboxVC model.

        Imports are deferred to here so that import errors surface at
        construction time with a clear message, not at module import.
        """
        try:
            from chatterbox.vc import ChatterboxVC
        except ImportError:
            raise ImportError(
                "Could not import chatterbox. Install it with:\n"
                "  pip install chatterbox-tts\n"
                "Or from source:\n"
                "  git clone https://github.com/resemble-ai/chatterbox\n"
                "  cd chatterbox && pip install -e ."
            )

        print(f"Loading ChatterboxVC on {self.device}...", flush=True)
        t0 = time.time()

        if self._model_path:
            self._model = ChatterboxVC.from_local(self._model_path, self.device)
        else:
            self._model = ChatterboxVC.from_pretrained(self.device)

        elapsed = time.time() - t0
        print(f"ChatterboxVC loaded in {elapsed:.1f}s", flush=True)

    @property
    def sample_rate(self) -> int:
        """Output sample rate of the model (always 24000 Hz)."""
        return OUTPUT_SAMPLE_RATE

    def set_target_voice(self, target_audio_path: str) -> None:
        """Pre-compute and cache target speaker embedding.

        Use this when converting multiple source files to the same target
        voice. Avoids redundant speaker embedding extraction on each call.

        Args:
            target_audio_path: Path to a WAV file of the target speaker.
                Should be 6-10 seconds of clean speech for best results.
                Longer audio is truncated to 10 seconds. Any sample rate
                is accepted (automatically resampled internally).

        Note:
            The speaker embedding captures vocal identity characteristics
            like timbre, pitch range, and speaking style. It is extracted
            by the CAMPPlus speaker encoder, producing an 80-dimensional
            x-vector representation.
        """
        self._model.set_target_voice(target_audio_path)

    def convert(
        self,
        source_audio: str,
        target_voice: str,
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """Convert source audio to sound like the target speaker.

        Args:
            source_audio: Path to the source WAV file. This is the speech
                whose content will be preserved but whose speaker identity
                will be replaced. Any sample rate is accepted.
            target_voice: Path to the target speaker's WAV file. The output
                will sound like this speaker. Should be 6-10 seconds of
                clean speech. Any sample rate is accepted.
            output_path: If provided, save the output WAV to this path.
                Parent directories are created automatically.

        Returns:
            NumPy array of the converted audio waveform at 24kHz, float32,
            shape ``(num_samples,)``.

        Raises:
            FileNotFoundError: If source_audio or target_voice doesn't exist.
            RuntimeError: If the model fails during inference.

        Example:
            >>> vc = VoiceConverter(device="cuda:0")
            >>> wav = vc.convert("speech.wav", "target.wav", "output.wav")
            >>> print(f"Output: {len(wav)} samples at {vc.sample_rate}Hz")
        """
        if not os.path.exists(source_audio):
            raise FileNotFoundError(f"Source audio not found: {source_audio}")
        if not os.path.exists(target_voice):
            raise FileNotFoundError(f"Target voice not found: {target_voice}")

        # Run voice conversion. The model internally:
        # 1. Loads target voice at 24kHz, extracts speaker embedding
        # 2. Loads source audio at 16kHz, tokenizes to S3 content tokens
        # 3. Runs flow-matching decoder (10 CFM steps) conditioned on speaker
        # 4. Converts mel-spectrogram to waveform via HiFi-GAN
        # 5. Applies PerTH watermark
        wav_tensor = self._model.generate(
            audio=source_audio,
            target_voice_path=target_voice,
        )

        # Convert from tensor [1, samples] to numpy [samples]
        wav_np = wav_tensor.squeeze().cpu().numpy().astype(np.float32)

        if output_path:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            sf.write(output_path, wav_np, self.sample_rate)

        return wav_np

    def convert_batch(
        self,
        source_paths: list[str],
        target_voice: str,
        output_dir: str,
    ) -> list[dict]:
        """Convert multiple source files to the same target voice.

        More efficient than calling convert() in a loop because the target
        speaker embedding is extracted only once.

        Args:
            source_paths: List of paths to source WAV files.
            target_voice: Path to the target speaker's WAV file.
            output_dir: Directory to write output files. Each output file
                is named ``{original_stem}_converted.wav``.

        Returns:
            List of result dicts, one per source file::

                {
                    "source": "/path/to/source.wav",
                    "output": "/path/to/output_converted.wav",
                    "duration_s": 3.45,
                    "elapsed_s": 1.23,
                    "success": True,
                }

            Failed conversions have ``success=False`` and an ``error`` key.

        Example:
            >>> vc = VoiceConverter(device="cuda:0")
            >>> results = vc.convert_batch(
            ...     ["s1.wav", "s2.wav", "s3.wav"],
            ...     "target.wav",
            ...     "output/",
            ... )
            >>> for r in results:
            ...     print(f"{r['source']} -> {r['output']} ({r['elapsed_s']:.1f}s)")
        """
        os.makedirs(output_dir, exist_ok=True)

        # Pre-compute target speaker embedding once
        self.set_target_voice(target_voice)

        results = []
        for source in source_paths:
            stem = Path(source).stem
            out_path = os.path.join(output_dir, f"{stem}_converted.wav")

            t0 = time.time()
            try:
                # Use generate() without target_voice_path since we already
                # set the target via set_target_voice() above.
                wav_tensor = self._model.generate(audio=source)
                wav_np = wav_tensor.squeeze().cpu().numpy().astype(np.float32)
                sf.write(out_path, wav_np, self.sample_rate)

                elapsed = time.time() - t0
                duration = len(wav_np) / self.sample_rate
                results.append({
                    "source": source,
                    "output": out_path,
                    "duration_s": round(duration, 2),
                    "elapsed_s": round(elapsed, 2),
                    "success": True,
                })
            except Exception as e:
                results.append({
                    "source": source,
                    "output": out_path,
                    "elapsed_s": round(time.time() - t0, 2),
                    "success": False,
                    "error": str(e),
                })

        return results
