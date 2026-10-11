# Humaneness Voice Converter · measured inference speed

This page separates **candidate generation**, **Best-of-N ranking**, and
**first-use preparation**. Mixing these timings would overstate the speed of a
service. The exact benchmark code is `research/benchmark_speed.py`; the
user-facing batched and ranked CLI is `infer.py`.

## Hardware and measured scope

- One NVIDIA GH200 120 GB on JUPITER, PyTorch 2.9.1+cu128, Chatterbox 0.1.7,
  10 flow steps, complete Humaneness Voice Converter checkpoint, target B fixed.
- Three post-warm-up repetitions for each setting; table shows the median
  wall time. The source had 4.27, 6.53 or 14.78 seconds of speech.
- A row generates **N stochastic takes of the same source/target pair** in a
  true shared GPU forward. This is not a batch of N different user requests.
- Source S3 tokens, score conditioning and reference embedding/acoustic prefix
  were ready before timing. Timed work: flow decoder, HiFT vocoder, GPU sync,
  and waveform transfer to host. Excluded: model load, raw-audio I/O,
  tokenization, Ears/CAMPPlus ranking, MP3 encoding and network.
- FP32 means no autocast; BF16 means PyTorch CUDA BF16 autocast around flow and
  HiFT. The weights are not permanently downcast for either mode. Peak GPU
  allocations in the BF16 tests were about 1.1–3.5 GiB, increasing with N and
  input duration; this is not total process/device usage.

| Source | Precision | N=1 | N=4 | N=8 | N=16 |
| --- | --- | ---: | ---: | ---: | ---: |
| 4.27 s | FP32 | 0.450 s | 0.487 s | 0.614 s | 1.079 s |
| 4.27 s | BF16 | 0.627 s | 0.627 s | 0.629 s | 0.650 s |
| 6.53 s | FP32 | 0.468 s | 0.526 s | 0.831 s | 1.534 s |
| 6.53 s | BF16 | 0.558 s | 0.605 s | 0.643 s | 0.665 s |
| 14.78 s | FP32 | 0.513 s | 0.782 s | 1.479 s | 2.892 s |
| 14.78 s | BF16 | 0.586 s | 0.628 s | 0.714 s | 1.116 s |

BF16 is **not universally faster** on this hardware: FP32 is faster for one
candidate and most four-candidate short/medium clips. BF16 wins clearly for
larger pools. The CLI retains BF16 as its default because that is the
precision used by the evaluated 40k generation path; `--precision fp32` is
available for latency experiments, but its output quality has not yet been
audited head-to-head. On the 6.53-s example, BF16 N=16 produces 16 candidate streams
in 0.665 s; that is 156.9 aggregate candidate-audio seconds per wall second,
**not** a 156.9× real-time *winner stream*. A single selected winner has
generation latency of 0.665 s, then needs ranking and delivery.

The separate original 40-case Best-of-N evaluation generated each candidate
**serially**, and scored each one. Across its 80 source/target pairs, mean
generation-plus-ranking time was about 0.73 s at N=1, 2.69 s at N=4, 5.31 s
at N=8, and 10.54 s at N=16. Its sources had mixed lengths; these numbers
must not be subtracted from a single-row batch timing as if the inputs and
measurement scope were identical. The equal-third, fixed-calibration selection
reward rose from 0.322 to 0.366/0.387/0.399; CE changed only from 6.449 to
6.459, while speaker cosine rose 0.755→0.776 and top-two-emotion MSE fell
0.124→0.074. The reward uses the same Ears model family used in training,
so it is not an independent human-perceived quality result.

## Tested batched/ranked CLI

`infer.py` now accepts `--candidates 1|4|8|16` and `--microbatch 1|2|4|8|16`.
It tokenizes the source once, prepares target and score conditioning once,
generates independent flow noise rows in true GPU microbatches, scores the
output with frozen Humaneness Ears Medium and CAMPPlus, then writes only the
selected 24-kHz WAV. The Best-of-N reward is the equal-third standardized
combination of CE, target cosine and negative source-top-two-emotion MSE using
the included fixed `conditioning/reward_calibration.json`. `--metrics` writes
the complete candidate proxy scores and timing to one JSON file. A fixed seed
**and** microbatch size define a reproducible run; changing microbatch
composition changes which RNG draws become which candidate.

On a 6.53-s input, a completed four-candidate `--microbatch 4 --precision
bf16` smoke produced a finite 24-kHz WAV with generation 0.616 s, ranking
0.135 s and preparation 6.351 s, for 7.126 s first-use total. The preparation
includes model load, source/target score prediction, source tokenization and
reference setup. A 16-candidate `--microbatch 8` smoke took 1.387 s generation,
0.206 s ranking and 6.206 s preparation (7.836 s total). The two eight-take
microbatches are intentionally slower than the single-batch N=16 decoder
timing above. A separate completed `--microbatch 16` smoke took 0.752 s for
generation, 0.261 s for ranking and 6.283 s for preparation (7.337 s
first-use total). Its generated output was finite and 24 kHz. A single
16-take batch fits a GH200 in this tested setup, but smaller GPUs may need a
lower microbatch. The ranking scores and chosen take change when the RNG
microbatch grouping changes.

For repeated requests, keep model, Ears and target conditioning resident in a
process. The tested CLI reloads them on each invocation; it is therefore a
reference command, not an optimized persistent server. It does not yet
implement multi-request batching, source-token caching across unrelated
requests, compiled kernels, async scoring or streamed output.

The [upstream Chatterbox VC BF16/batching recipe](https://github.com/LAION-AI/chatterbox-voice-conversion#speed-optimization-bf16-flow-and-true-gpu-batching)
uses an RTX 3090 and Vanilla S3Gen; its numbers are separate from the GH200
Humaneness Voice Converter results here.
