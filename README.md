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

For higher GPU throughput, see [BF16 flow + true GPU batching](#speed-optimization-bf16-flow-and-true-gpu-batching).
These are explicit advanced settings, not automatic defaults of `VoiceConverter`.

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

This convenience API currently loops over files; it is **not** a simultaneous
GPU batch. S3Gen supports true GPU batching, demonstrated by
[`convert_n()`](./expressive_bestofn/scripts/vc_batch.py) and the recipe below.

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
| 10 | Good (default) | Hardware/precision dependent; see measured throughput below | Production use |
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

4. **Cache the target, then use real GPU batching when possible.** `convert_batch()` and `set_target_voice()` avoid redundant target encoding; `convert_n()` batches candidate synthesis. BF16 flow is the next useful speed optimization; see the measured recipe below.

5. **GPU memory**: The model requires ~4 GB of GPU memory. If you have multiple GPUs, use the `device` parameter to spread load.

## Speed optimization: BF16 flow and true GPU batching

### What was actually measured

An October 9, 2026 Humaneness Small Playground integration tested S3Gen voice
conversion on an **RTX 3090**, independently of the older expressive Best-of-N
study below. The tested S3Gen has **263,933,767 parameters**. Keeping the original
**10 flow steps and PerTH watermark**, the biggest gain came from **BF16 only in
the flow estimator**, combined with GPU microbatches up to eight. Tokenizer,
conditioning encoder and vocoder stayed FP32. This is not fewer denoising steps,
not whole-model FP16, and not FP8.

Measured environment: Python 3.12.13, PyTorch/torchaudio 2.6.0+cu124, CUDA 12.4,
`chatterbox-tts` 0.1.7, `s3tokenizer` 0.3.0, `diffusers` 0.29.0. Checkpoint:
`ResembleAI/chatterbox` revision `5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18`,
`s3gen.safetensors` SHA-256
`2b78103c654207393955e4900aac14a12de8ef25f4b09424f1ef91941f161d4e`.

The numbers below are **median wall times of three warmed repetitions** with
one fixed target recording and controlled equal-length source windows from
existing TTS-study audio. Target conditioning was prepared before timing.
Timed stages include source upload/resampling, S3 tokenization, flow, vocoder,
output download, watermark and output resampling to the demo's 48 kHz pipeline.
They exclude target preparation/model loading, TTS generation, reward scoring,
alignment, MP3 encoding and network. They are **not kernel-only timings** or
end-to-end service latency. Native VC output remains 24 kHz.

| Input per candidate | Candidates N | FP32 serial | FP32 GPU batch | BF16 flow GPU batch | Speedup vs FP32 serial |
|---|---:|---:|---:|---:|---:|
| 1.6 s | 1 | 0.341 s | 0.340 s | 0.308 s | 1.1× |
| 1.6 s | 4 | 1.364 s | 1.059 s | 0.386 s | 3.5× |
| 1.6 s | 8 | 2.714 s | 2.004 s | 0.702 s | 3.9× |
| 1.6 s | 16 | 5.434 s | 4.013 s | 1.408 s | 3.9× |
| 3.2 s context window | 1 | 0.453 s | 0.453 s | 0.316 s | 1.4× |
| 3.2 s context window | 4 | 1.812 s | 1.304 s | 0.517 s | 3.5× |
| 3.2 s context window | 8 | 3.622 s | 2.425 s | 0.943 s | 3.8× |
| 3.2 s context window | 16 | 7.235 s | 4.841 s | 1.886 s | 3.8× |

GPU batch columns use microbatch `min(N, 8)`; N=16 therefore uses two batches.
For 1.6 s inputs, BF16 aggregate throughput is **18.2 audio-seconds per second**
at N=8 or 16. That counts all candidates, not just one eventual winner.

For steady streaming, a 3.2 s window consists of **1.6 s previous context +
1.6 s new audio**. Counting only the new audio gives:

| N | BF16 time per round | Round RTF: wall / 1.6 s | New candidate audio / wall second |
|---:|---:|---:|---:|
| 1 | 0.316 s | 0.197 | 5.1 s/s |
| 4 | 0.517 s | 0.323 | 12.4 s/s |
| 8 | 0.943 s | 0.590 | 13.6 s/s |
| 16 | 1.886 s | 1.179 | 13.6 s/s |

**Eight candidate conversions fit the 1.6 s interval for VC alone; sixteen do
not on this card/configuration.** Do not count re-rendered context as newly
produced audio, or divide latency by N and claim the winner stream is real-time.
TTS and scoring still consume additional time; these are candidate streams
within one search job, not independently scheduled concurrent user requests.

### Apply the BF16 speedup with the existing batched helper

Run from this repository root with the existing dependencies installed. This
uses private Chatterbox model attributes; it was tested against version 0.1.7
and should be rechecked after dependency upgrades. The example uses whatever
checkpoint the normal loader resolves; pin/verify the revision above for a
strict benchmark reproduction.

```python
from copy import deepcopy
import soundfile as sf
import torch
from chatterbox_vc import VoiceConverter
from expressive_bestofn.scripts.vc_batch import convert_n

converter = VoiceConverter(device="cuda:0")
converter.set_target_voice("target.wav")  # encode once, reuse for all candidates
model = converter._model
fp32_estimator = model.s3gen.flow.decoder.estimator
model.s3gen.flow.decoder.estimator = deepcopy(fp32_estimator).to(
    dtype=torch.bfloat16
).eval()
try:
    # The helper tokenizes ONE source once and repeats its tokens for N draws.
    convert_n(model, "source.wav", n=8, seed=1234)  # warm-up
    waves, flow_vocoder_seconds, seconds_per_candidate = convert_n(
        model, "source.wav", n=8, seed=1234
    )
finally:
    model.s3gen.flow.decoder.estimator = fp32_estimator  # untouched FP32 copy

for i, wave in enumerate(waves):
    marked = model.watermarker.apply_watermark(wave, sample_rate=model.sr)
    sf.write(f"converted-{i:02d}.wav", marked, model.sr)
```

The helper's `flow_vocoder_seconds` excludes tokenization, watermark and file
I/O; it must not be compared directly with the full-stage table above. Eight
outputs are different stochastic conversions of the **same** source, not
eight arbitrary source files. The Playground also demonstrated batched
conversion of distinct equal-length candidate chunks; that streaming wrapper
is application code, **not a new public `VoiceConverter` method** in this repo.
Do not cast back from BF16 and assume the original FP32 weights were restored;
retain/reload the original weights for honest A/B comparisons.

Implementation lessons for batching distinct sources:

- Group by actual window length and bound microbatch size; unlike lengths were
  kept separate rather than silently padding acoustic contexts. No speedup is
  claimed for an arbitrary mixed-duration offline file list.
- Cache resampling kernels and fixed-target conditioning. Tokenize distinct
  sources in a batch; vectorize equal-length STFT/log-mel extraction while
  preserving the original **per-wave** log-mel floor, not a batch-wide maximum.
- Use independent per-candidate diffusion seeds if rows must retain their seed
  when batch composition changes. The existing `convert_n()` uses one batch
  RNG seed; it does not expose the Playground's per-row seed implementation.
  Protect the upstream TTS RNG when inserting VC into an autoregressive stream.
- Keep an FP32 comparison path and test finite audio, duration, content and
  identity. Neither batch-vs-serial nor BF16-vs-FP32 waveforms are bit-exact.

### Streaming, quality checks and unsuccessful optimizations

The integration demonstrated **windowed streaming**, not a claim that the
ordinary file API is a native causal streaming decoder: retain 1.6 s source
context, buffer a minimum initial 0.8 s, and hold a 40 ms converted tail for a
crossfade with the re-rendered **old** samples. Flush pending input and held
tail at EOS. Crossing the seam must not repeatedly shorten or duplicate speech;
check duration drift and retain separate pending/context/tail state for each
branch. This seam recipe is a measured starting point, not an artifact-free
guarantee for every voice and emotion.

In live chunk search, **convert candidates before the rewards**, then stream
only committed converted PCM. A top-2 beam may wait for a common converted
prefix: shared TTS tokens do not prove identical stochastic VC waveforms.
Ordinary complete-take Best-of-N still waits for all takes before selecting
a winner. Rank-first/convert-only-the-winner is a different quality objective,
not equivalent to judging converted candidates.

Real HTTPS four-candidate search tests delivered the first converted MP3 chunk
after **2.3–2.5 s**, including TTS/search overhead. First VC use took longer to
load the model; a separate cold TTS compilation took about 39 s. Those are
individual integration checks, not guaranteed latencies or VC-only benchmarks.

Eight short (1.6 s) clips converted to one target produced these mean proxies:

| Mode | ECAPA cosine | Ears Medium Content Enjoyment | Production Quality | DNSMOS P.808 proxy |
|---|---:|---:|---:|---:|
| FP32 serial | 0.5741 | 5.435 | 7.094 | 3.111 |
| FP32 batch | 0.5745 | 5.437 | 7.095 | 3.110 |
| BF16 flow batch | 0.5715 | 5.437 | 7.091 | 3.109 |

This small numerical-optimization pilot found nearly unchanged means, **not**
universal speaker/emotion preservation or human MOS. Full-transcript WER is not
meaningful for these truncated benchmark inputs. Recheck long clips, bursts,
whispering, intense emotions and many independent references before a broad
quality claim. Streaming/windowing itself can change timbre or emotional delivery.

Other measured options: TF32 reduced an eight-clip 1.6 s batch from 2.022 to
1.680 s in a separate run, less than BF16 flow's gain. Two full FP32 model
copies/two CUDA streams gave 1.887 s at N=8 versus 2.004 s for one batched copy
(roughly 6% faster, at extra memory/scheduling cost); no replica speedup is
established on top of BF16. Dynamic `torch.compile` of the flow estimator failed
with `integer division or modulo by zero` in the tested stack; compilation was
not deployed. Static-shape compilation, CUDA graphs and removing host-sync
graph breaks are sensible **future experiments**, not measured speedups here.

[English benchmark with raw timing JSON and audio comparisons](https://switch-surprised-clock-sparc.trycloudflare.com/vc-performance/)
(temporary Playground tunnel; the key numbers and reproduction recipe are
retained in this README).

### RTX 4090, RTX 5090 and H100: expectations, not measurements

**Only RTX 3090 was benchmarked here.** The same BF16-flow/batching recipe is a
good first baseline on newer cards, but kernel support, CPU overhead, sequence
length, reference duration and batch size prevent reliable extrapolation from
advertised Tensor FLOPS. Do not label the 3090 numbers as 4090 results.

- **RTX 4090 (Ada, 24 GB):** expect more headroom for compute-heavy BF16 batches,
  not a fixed multiple of the entire pipeline. Memory capacity is still 24 GB,
  so larger batches are not automatically safe. Measure N=1/4/8/16 and the
  complete per-round latency; single-stream CPU/launch overhead may dominate.
- **RTX 5090 (Blackwell, 32 GB):** potentially more throughput and room for
  larger batches, but use a PyTorch/CUDA/kernel stack supporting that device;
  the measured CUDA 12.4 stack is not a 5090 setup recommendation. Start with
  BF16, then test compatible FP8 kernels after quality checks.
- **H100 (Hopper):** BF16 batching is the baseline; hardware FP8 and Transformer
  Engine make calibrated FP8 linear/attention-heavy parts worth testing.
  Benchmark the actual PCIe/SXM configuration and latency, rather than copying
  an LLM training speedup claim. No H100 concurrency or RTF is established here.

**FP8 is exploratory, not a drop-in `.to(float8)` optimization.** Use supported
FP8 kernels/modules with appropriate activation/weight scaling and calibration
across references, lengths, emotions and denoising timesteps. Start with flow
GEMMs; keep sensitive normalization, reductions, output layers and the vocoder
in BF16/FP32 until validated. Current Transformer Engine FP8 linear paths have
shape/alignment constraints; masking or padding must not alter audio semantics.
Quantization/scaling overhead can outweigh savings for small batches. Ada,
Hopper and Blackwell have different capabilities and kernel requirements;
hardware support alone does not make this model's existing operations FP8-fast.

Hardware references: [NVIDIA Ada architecture / RTX 4090](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf),
[RTX 5090 specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5090/),
[H100](https://www.nvidia.com/en-sg/data-center/h100/), and
[Transformer Engine low-precision guidance](https://docs.nvidia.com/deeplearning/transformer-engine/examples/fp8_primer.html).

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
