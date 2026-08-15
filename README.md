# Chatterbox Voice Conversion

A Python library for zero-shot voice conversion built on [Resemble AI's Chatterbox](https://github.com/resemble-ai/chatterbox) S3Gen model. Converts the speaker identity of any speech audio to match a target speaker while preserving the original linguistic content and prosody.

## How It Works

Chatterbox VC uses a three-stage pipeline:

```
Source audio (any speaker)     ─→  S3Tokenizer (16kHz)  ─→  Content tokens (25 tokens/sec)
                                                                          ↓
Target speaker audio (6-10s)   ─→  CAMPPlus encoder     ─→  Speaker embedding (192-dim x-vector)
                                                                          ↓
                                                              S3Gen Flow-Matching Decoder
                                                              (Conditional Flow Matching, 10 steps)
                                                                          ↓
                                                              Mel-spectrogram (80 bins)
                                                                          ↓
                                                              HiFi-GAN Vocoder + F0 Predictor
                                                                          ↓
                                                              24kHz output waveform
                                                                          ↓
                                                              PerTH Watermark (imperceptible)
```

**Stage 1 — Content Extraction**: The S3Tokenizer encodes the source speech at 16kHz into discrete content tokens at 25Hz (one token per 40ms). These tokens capture *what* is said — phonetic content, prosody timing, and rhythm — but discard speaker identity information.

**Stage 2 — Speaker Conditioning**: The CAMPPlus speaker encoder extracts a 192-dimensional x-vector from the target speaker's reference audio. This embedding captures the target's vocal identity: timbre, pitch range, and speaking style.

**Stage 3 — Waveform Synthesis**: The S3Gen decoder takes the content tokens and speaker embedding, then uses Conditional Flow Matching (CFM) to iteratively refine noise into a mel-spectrogram over N timesteps (default 10). The mel-spectrogram is converted to a 24kHz waveform by the HiFi-GAN vocoder, which includes an F0 (pitch) predictor for natural intonation. Finally, an imperceptible PerTH watermark is embedded.

## Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.0+ with CUDA support (GPU strongly recommended — CPU inference is ~20x slower)
- ~4 GB GPU memory

### Install from source

```bash
git clone https://github.com/LAION-AI/chatterbox-voice-conversion.git
cd chatterbox-voice-conversion
pip install -e .
```

This installs the `chatterbox_vc` package and its dependencies, including `chatterbox-tts` which provides the underlying model.

### Install dependencies only

If you prefer to install dependencies manually:

```bash
pip install chatterbox-tts>=0.1.1 torch torchaudio librosa soundfile numpy
```

## Quick Start

### Python API

```python
from chatterbox_vc import VoiceConverter

# Load model (~1.5 GB, auto-downloaded from HuggingFace on first run)
vc = VoiceConverter(device="cuda:0")

# Convert source audio to sound like the target speaker
wav = vc.convert("source_speech.wav", "target_speaker.wav", "output.wav")

print(f"Output: {len(wav)} samples at {vc.sample_rate}Hz")
# Output: 72000 samples at 24000Hz
```

### Command-line

```bash
# Single file conversion
python examples/basic_conversion.py \
    --source source_speech.wav \
    --target target_speaker.wav \
    --output converted.wav \
    --device cuda:0

# Batch conversion (extracts target speaker embedding once)
python examples/batch_conversion.py \
    --sources audio1.wav audio2.wav audio3.wav \
    --target target_speaker.wav \
    --output-dir ./converted/ \
    --device cuda:0
```

## API Reference

### `VoiceConverter`

```python
VoiceConverter(device="cuda:0", model_path=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `device` | `str` | `"cuda:0"` | PyTorch device. Use `"cuda:0"`, `"cuda:1"`, etc. for GPU, or `"cpu"` (very slow). |
| `model_path` | `str` or `None` | `None` | Path to a local checkpoint directory. If `None`, weights are auto-downloaded from HuggingFace (`ResembleAI/chatterbox`). |

#### `convert(source_audio, target_voice, output_path=None)`

Convert a single audio file.

| Parameter | Type | Description |
|-----------|------|-------------|
| `source_audio` | `str` | Path to the source WAV file (speech to convert). Any sample rate. |
| `target_voice` | `str` | Path to the target speaker's WAV file (6-10s of clean speech recommended). Any sample rate. |
| `output_path` | `str` or `None` | If provided, saves the output WAV to this path. Directories are created automatically. |

**Returns**: NumPy array of float32 audio at 24kHz, shape `(num_samples,)`.

#### `convert_batch(source_paths, target_voice, output_dir)`

Convert multiple source files to the same target voice. More efficient than calling `convert()` in a loop because the target speaker embedding is extracted only once.

| Parameter | Type | Description |
|-----------|------|-------------|
| `source_paths` | `list[str]` | List of source WAV file paths. |
| `target_voice` | `str` | Path to the target speaker's WAV file. |
| `output_dir` | `str` | Directory for output files (named `{stem}_converted.wav`). |

**Returns**: List of result dicts with keys: `source`, `output`, `duration_s`, `elapsed_s`, `success`, and `error` (on failure).

#### `set_target_voice(target_audio_path)`

Pre-compute and cache the target speaker embedding. Useful when converting many files to the same target voice.

#### `sample_rate`

Property returning the output sample rate (always `24000`).

## Subprocess Worker

For production deployments where you need GPU isolation or a separate Python environment, use the subprocess worker:

```python
from chatterbox_vc.worker import WorkerClient

# Spawn a worker subprocess on a specific GPU
client = WorkerClient(device="cuda:1")

# Convert audio (synchronous)
result = client.convert("source.wav", "target.wav", "output.wav")
print(f"Done in {result['elapsed']}s at {result['sample_rate']}Hz")

# Clean up
client.shutdown()
```

Or run the worker directly and communicate via JSON-line protocol:

```bash
python -m chatterbox_vc.worker --device cuda:0
```

The worker reads JSON requests from stdin and writes responses to fd 3 (or stdout if fd 3 is unavailable):

```json
{"source": "/path/to/source.wav", "target": "/path/to/target.wav", "output": "/path/to/output.wav"}
```

Response:
```json
{"status": "ok", "output": "/path/to/output.wav", "sample_rate": 24000, "elapsed": 2.34}
```

## Tunable Hyperparameters

The Chatterbox VC model has several hyperparameters that affect output quality, speed, and characteristics. Most are architectural constants, but a few can be tuned at inference time.

### Inference-Time Parameters

#### CFM Timesteps (`n_cfm_timesteps`)

**Default**: `10` · **Range**: `1–100` · **Location**: Passed to `model.generate()` internally

The number of Conditional Flow Matching (CFM) denoising steps. This is the most impactful quality/speed trade-off:

| Steps | Quality | Speed | Use Case |
|-------|---------|-------|----------|
| 1–5 | Lower, may have artifacts | Very fast | Real-time/streaming, previews |
| 10 | Good (default) | ~0.3x real-time on GPU | Production use |
| 20–50 | Marginally better | 2–5x slower | Maximum quality, offline processing |
| 50+ | Diminishing returns | Very slow | Not recommended |

To modify: Edit the `generate()` call in `chatterbox/vc.py` or patch the model's `inference()` method.

#### Classifier-Free Guidance Rate (`inference_cfg_rate`)

**Default**: `0.7` · **Range**: `0.0–2.0` · **Location**: `chatterbox/models/s3gen/configs.py`

Controls how strongly the model follows the speaker conditioning signal:

| Value | Effect |
|-------|--------|
| 0.0 | No guidance — output may not match target speaker well |
| 0.5 | Light guidance — more natural but less speaker-similar |
| 0.7 | Default — good balance of naturalness and speaker match |
| 1.0+ | Strong guidance — closer speaker match but may sound less natural |

Formula: `output = (1 + cfg_rate) * conditioned - cfg_rate * unconditioned`

#### Temperature

**Default**: `1.0` · **Range**: `0.1–2.0` · **Location**: `chatterbox/models/s3gen/flow_matching.py`

Scales the initial noise fed to the flow-matching decoder:

| Value | Effect |
|-------|--------|
| < 1.0 | More deterministic, potentially less expressive |
| 1.0 | Default stochasticity |
| > 1.0 | More variation, potentially more expressive but less stable |

### Reference Audio Parameters

#### Target Speaker Reference Length

**Encoding**: 6 seconds @ 16kHz (`ENC_COND_LEN = 96,000 samples`)
**Decoding**: 10 seconds @ 24kHz (`DEC_COND_LEN = 240,000 samples`)

**Location**: `chatterbox/vc.py`

The target reference audio is truncated to these lengths. Longer audio is clipped; shorter audio is used as-is.

**Recommendations**:
- **Minimum**: 3 seconds (shorter clips produce less stable speaker embeddings)
- **Optimal**: 6–10 seconds of clean, single-speaker speech
- **Content**: Use diverse phonetic content (not just one word repeated)
- **Quality**: Clean recording, minimal background noise, no music

#### Source Audio

No explicit length limit on source audio, but:
- Very short clips (< 1 second) may produce artifacts
- Very long clips (> 60 seconds) work but increase processing time linearly
- Any sample rate is accepted (automatically resampled to 16kHz internally)

### Architecture Constants (Advanced)

These require model retraining to change but are documented for understanding:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Output sample rate | 24,000 Hz | Fixed by HiFi-GAN vocoder architecture |
| Mel bins | 80 | Mel-spectrogram resolution |
| Mel hop size | 480 samples (20ms) | Temporal resolution of mel frames |
| Mel frequency range | 0–8,000 Hz | Captured frequency band |
| Content token rate | 25 tokens/sec | S3Tokenizer output rate |
| Content vocabulary | 6,561 tokens | Discrete codebook size (3^8) |
| Speaker embedding dim | 192 | CAMPPlus x-vector output |
| Conformer blocks | 6 | Encoder depth |
| Decoder mid-blocks | 12 | UNet1D bottleneck depth |
| Upsample rates | 8 × 5 × 3 = 120x | Mel-to-waveform upsampling |
| Time scheduler | Cosine | CFM step distribution: `t = 1 - cos(t × π/2)` |

## Tips for Best Results

1. **Target reference quality matters most.** Use 6-10 seconds of clean speech with varied phonetic content. Background noise in the reference degrades all conversions.

2. **Source audio should be intelligible.** The model preserves content from the source — if the source is unclear, the output will be too.

3. **Same-language works best.** Cross-language conversion (e.g., English source → Japanese target speaker) may produce accented output.

4. **Batch conversion is faster.** When converting multiple files to the same target voice, use `convert_batch()` or `set_target_voice()` to avoid redundant speaker embedding extraction.

5. **GPU memory**: The model requires ~4 GB of GPU memory. If you have multiple GPUs, use the `device` parameter to spread load.

## Expressive Best-of-N conversion (quality × emotion ranking)

Voice conversion is stochastic — the flow-matching decoder starts from Gaussian noise, so every
run is a different draw. The [`expressive_bestofn/`](./expressive_bestofn) add-on turns that into
a strength: it **generates many candidates in one batched pass, scores each for emotion, quality
and speaker/timbre similarity, optionally restores them with SIDON, and ranks them** so the
result is *as clean as possible while the emotion of the original performance is preserved*.

**▶️ Live demo:** https://tts-agi-chatterbox-expressive-bestofn.static.hf.space &nbsp;·&nbsp;
**📄 Method:** [`expressive_bestofn/METHOD.md`](./expressive_bestofn/METHOD.md)

How it works, in short:

- **Batched multi-seed generation** — repeat the source content tokens along the batch dim so N
  candidates come out of a single forward pass (~180 ms/candidate at N=32 vs ~530 ms at N=1).
- **Scoring** — a frozen [BUD-E-Whisper](https://huggingface.co/laion/BUD-E-Whisper) encoder feeds
  40 emotion experts ([Empathic-Insight-Voice-Small](https://huggingface.co/laion/Empathic-Insight-Voice-Small))
  and DNSMOS-distilled quality experts
  ([Empathic-Insight-Voice-Plus](https://huggingface.co/laion/Empathic-Insight-Voice-Plus));
  speaker/timbre similarity to the target uses
  [ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) and the
  [Orange Speaker-wavLM-tbr](https://huggingface.co/Orange/Speaker-wavLM-tbr) timbre embedding.
- **SIDON restoration** — optional 48 kHz enhancement
  ([sarulab-speech/Sidon](https://github.com/sarulab-speech/Sidon),
  [sidon-v0.1](https://huggingface.co/sarulab-speech/sidon-v0.1)).
- **Reward** — `z(peak-emotion) + z(Overall-Quality)`, so the winning take is expressive **and**
  clean. Best-of-N mostly buys cleaner audio at equal emotion; the sweet spot is **N≈8**
  (~496 GPU-hours per 1M samples with a rank-then-SIDON-the-winner pipeline).

See [`expressive_bestofn/README.md`](./expressive_bestofn/README.md) for the full reproduce steps.

## Project Structure

```
chatterbox-voice-conversion/
├── chatterbox_vc/
│   ├── __init__.py          # Package entry point, exports VoiceConverter
│   ├── convert.py           # Core VoiceConverter class with convert/batch APIs
│   └── worker.py            # Subprocess worker + WorkerClient for GPU isolation
├── examples/
│   ├── basic_conversion.py  # Single-file conversion CLI example
│   └── batch_conversion.py  # Multi-file batch conversion CLI example
├── expressive_bestofn/      # Best-of-N + scoring + SIDON + reward ranking (add-on)
│   ├── README.md            # Overview + reproduce steps
│   ├── METHOD.md            # Detailed what/why/how write-up
│   ├── scripts/             # The exact pipeline (generate, score, refine, reward, analysis)
│   ├── results/             # Computed analysis JSON (best-of-k, SIDON effect, reward, GPU-hours)
│   └── demo/index.html      # Reference copy of the live demo page
├── tests/
│   └── test_conversion.py   # Integration test that verifies end-to-end conversion
├── LICENSE                  # Apache 2.0
├── pyproject.toml           # Package metadata and dependencies
└── README.md                # This file
```

## License

This project is licensed under the [Apache License 2.0](LICENSE).

The underlying Chatterbox model is developed by [Resemble AI](https://github.com/resemble-ai/chatterbox) and has its own license terms.

## Acknowledgments

- [Resemble AI](https://www.resemble.ai/) for the Chatterbox TTS/VC model
- [LAION](https://laion.ai/) for supporting open-source AI research
