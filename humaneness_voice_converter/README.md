# Humaneness Voice Converter · 40k full-flow checkpoint

This directory accompanies the [complete model checkpoint](https://huggingface.co/laion/Humaneness-Voice-Converter) and the [standalone demo](https://huggingface.co/spaces/laion/Humaneness-Voice-Converter-Demo). It is an expressive **speech-to-speech voice converter**, not a text-to-speech model. Give it source speech and a 5–10-second target-voice recording. The S3 tokenizer supplies 25 content tokens per second; the target supplies a 192-dimensional CAMPPlus identity vector and acoustic prefix. Frozen Humaneness Ears Medium estimates source emotion/style and target stable-timbre attributes; the trained score projection conditions a 10-step Chatterbox flow decoder and HiFT vocoder.

The released weights start from upstream `ResembleAI/chatterbox` S3Gen, then a 300.017-hour rank-128 score-conditioned SFT adapter, group-relative flow RL, a 24k-group Speaker-Mix adapter (40% Content Enjoyment, 40% top-two-emotion match, 20% target-speaker similarity), and finally a **40k-group full-flow continuation** after merging the adapter. The last stage optimizes 112,620,864 flow parameters with a 2k-group warm-up and one cosine schedule from 3e-7 to 3e-8, using groups of 16 on four JUPITER GH200 GPUs. The tokenizer, CAMPPlus, HiFT and Ears predictor stay frozen.

## Inference and reproducibility

`infer.py` is the raw-audio CLI actually smoke-tested against the complete exported SafeTensors checkpoint. The model repository contains the 1.0-GB `model.safetensors`, frozen Ears predictor, score order and proof of export. Download it and run:

```bash
hf download laion/Humaneness-Voice-Converter --local-dir humaneness-vc
python humaneness_voice_converter/infer.py \
  --model-dir humaneness-vc --source source.wav \
  --target consenting_target.wav --output converted.wav
```

The JUPITER-tested Python 3.13/PyTorch 2.9.1 dependency snapshot is in `requirements.txt`. Chatterbox 0.1.7 has older metadata pins for Python 3.13; install the listed packages first, then `chatterbox-tts==0.1.7 --no-deps`, as detailed in the model card. This is a tested *existing environment*, not a promise that a fresh resolver run works on all platforms.

`REPRODUCING.md` records the precise training data selection, lineage, reward, optimizer/checkpoint schedule, and limitations. `training/` contains the research source snapshots and Slurm scripts; their Scratch paths must be adapted to an equivalent packed data layout. The `research/` scripts recreate the local evaluation and benchmark protocol, not an inference service. Never unpack the corpus into millions of clip files.

## Measured listening-set scores

The current demo contains 40 source cases × two target voices; 32 sources are English and eight German. These are *curated, partly training-overlapping* cases, not an independent blinded human benchmark. CE/PQ/emotion are Humaneness Ears Medium predictions, and target-speaker cosine is CAMPPlus. Higher CE/PQ/cosine and lower emotion MSE are directionally preferable.

| Model or selection | CE ↑ | PQ ↑ | Emotion MSE ↓ | Target cosine ↑ |
| --- | ---: | ---: | ---: | ---: |
| Vanilla Chatterbox | 6.173 | 7.747 | 0.140 | 0.806 |
| Humaneness 40k, one take | 6.449 | 7.999 | 0.124 | 0.755 |
| Seed-VC v1 | 5.993 | 7.497 | 0.216 | 0.859 |
| Seed-VC v2 | 5.884 | 7.395 | 0.403 | 0.775 |
| X-VC (2026) | 5.795 | 7.319 | 0.239 | 0.712 |
| Humaneness Best-of-4 | 6.458 | 8.004 | 0.092 | 0.766 |
| Humaneness Best-of-8 | 6.458 | 8.002 | 0.080 | 0.772 |
| Humaneness Best-of-16 | 6.459 | 8.000 | 0.074 | 0.776 |

The 40k model raises these quality proxies versus Vanilla but has lower speaker cosine. Seed-VC v1 has higher target cosine than the 40k model. External arms used their official checkpoints on the same source/target pairs; see the demo for audio and provenance. This table is **not** a claim of universal superiority. There is no independent human MOS, WER or speaker-level disjointness result here.

## Best-of-N and speed

The Best-of-N score gives one third each to standardized AudioBox Content Enjoyment, target CAMPPlus cosine, and negative mean squared error on the source's strongest two emotion axes. Standardization is fixed from 93 noncurated source cases; all N share nested candidate pools. The earlier **sequential** full evaluation generated and scored each candidate separately: around 0.73 s for N=1, 2.69 s for N=4, 5.31 s for N=8 and 10.54 s for N=16 per source/target pair. That is approximately linear growth in per-pair latency, though the selected proxy reward improves from 0.322 to 0.366/0.387/0.399. The reward must be nondecreasing for nested pools; individual quality axes need not be.

True GPU batching changes the generator cost: one source tokenization and target condition are reused while independent Gaussian noise rows yield N stochastic takes in a shared forward pass. On a GH200, our 10-step 40k decoder plus HiFT, **excluding** source tokenization, Ears/CAMPPlus ranking, MP3 encoding, model load and network, was measured with three warmed repetitions. For a 6.53-s source, median FP32 wall time was 0.468/0.526/0.831/1.534 s for N=1/4/8/16; BF16 autocast was 0.558/0.605/0.643/0.665 s. BF16 is not uniformly faster: FP32 wins for single and small batches, while BF16 wins clearly for N=8/16. The per-candidate aggregate throughput improves sharply, but **one Best-of-16 winner still waits for the whole batch and its ranking**. These decoder-only numbers must not be quoted as end-to-end service latency. More representative durations and FP8 results are in `SPEED.md` when validated.

The upstream library has its own [BF16/batching recipe](../README.md#speed-optimization-bf16-flow-and-true-gpu-batching), measured on an RTX 3090 with Vanilla S3Gen; our GH200/40k figures are distinct. `research/benchmark_speed.py` documents our exact 40k test. FP8 is experimental: the supported quantizer covers only estimator `Linear` modules, not the largely convolutional flow, and requires compatibility changes to upstream `inference_mode`. We do not recommend FP8 without a successful quality-and-speed result.

## Rights and responsible use

This GitHub repository's source license is Apache-2.0; the model card states CC BY 4.0 for LAION-authored weights/documentation with upstream Chatterbox MIT attribution retained. Source recordings and external models keep their own rights and licenses. Use only audio/voices you are authorized to use. Research and consented dataset augmentation are the intended use; impersonation and deceptive use are discouraged. The converter can alter words, emotion, quality or perceived identity—test it before any real-world deployment.
