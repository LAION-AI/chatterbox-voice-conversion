"""S3 V2 waveform-window adapter; leaves vendor code and learned weights intact.

Vendor long-mel splitting can append a redundant final window shorter than the
half-overlap trimming. Use bounded 29-second windows, stop as soon as the final
window covers the waveform, and merge four-second overlaps at their midpoint.
"""
import numpy as np

def tokenize_waveform(tokenizer, wave, sample_rate=16000):
    assert sample_rate == 16000 and wave.ndim == 1
    window = 29 * sample_rate
    stride = 25 * sample_rate
    segments = []
    start = 0
    while start < len(wave):
        end = min(start + window, len(wave))
        tokens, lengths = tokenizer([wave[start:end]])
        segments.append(tokens[0, :int(lengths[0])].cpu().numpy().astype(np.int32))
        if end == len(wave):
            break
        start += stride
    assert segments
    return np.concatenate([
        codes[0 if i == 0 else 50:len(codes) if i == len(segments)-1 else -50]
        for i, codes in enumerate(segments)
    ])
