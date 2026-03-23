"""
Chatterbox Voice Conversion
============================

A voice conversion pipeline built on Resemble AI's Chatterbox S3Gen model.
Converts the speaker identity of any speech audio to match a target speaker,
while preserving the original linguistic content and prosody.

Architecture:
    Source audio -> S3Tokenizer (16kHz, content tokens)
                                                        -> S3Gen flow-matching decoder -> HiFi-GAN vocoder -> 24kHz output
    Target audio -> Speaker encoder (CAMPPlus x-vector)

Quick start:
    >>> from chatterbox_vc import VoiceConverter
    >>> vc = VoiceConverter(device="cuda:0")
    >>> vc.convert("source.wav", "target_speaker.wav", "output.wav")
"""

from chatterbox_vc.convert import VoiceConverter

__version__ = "0.1.0"
__all__ = ["VoiceConverter"]
