# Expressive Best-of-N Voice Conversion — Method

This document explains, in detail, **what** the pipeline does, **why** each step exists, and
**how** it is implemented. The goal is a single sentence:

> Convert speech to a target voice so that the result is **as clean as possible** while the
> **emotion of the original performance is preserved** — by generating many candidates,
> scoring them, and keeping the best.

Voice conversion (VC) is not deterministic: the flow-matching decoder starts from Gaussian
noise, so every run produces a slightly different take. Instead of accepting one random draw,
we **sample a batch of candidates**, measure each one, and **rank** them. This turns a
one-shot generator into a search over quality and expressivity.

---

## 0 · Ingredients

| Component | Repo | Role |
|---|---|---|
| Chatterbox VC (S3Gen) | this repo | speaker/timbre conversion, content preserved |
| Empathic-Insight-Voice-Plus | https://huggingface.co/laion/Empathic-Insight-Voice-Plus | 40 emotion experts + DNSMOS-distilled quality experts (on a frozen BUD-E-Whisper encoder) |
| Empathic-Insight-Voice-Small | https://huggingface.co/laion/Empathic-Insight-Voice-Small | the 40 emotion / 12 attribute experts (full-sequence MLPs) |
| BUD-E-Whisper | https://huggingface.co/laion/BUD-E-Whisper | frozen audio encoder feeding all experts |
| SIDON | https://github.com/sarulab-speech/Sidon · https://huggingface.co/sarulab-speech/sidon-v0.1 | speech restoration / enhancement post-processor (48 kHz) |
| ECAPA-TDNN | https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb | 192-d speaker embedding (speaker similarity) |
| Orange Speaker-wavLM-tbr | https://huggingface.co/Orange/Speaker-wavLM-tbr | 128-d **timbre** embedding (speaker similarity, second opinion) |

The experiment target voice is **"Measured Slavic Historian"** (`emolia_c0542`). The sources
are 52 clips from the [voice-profile page](https://tts-agi-moss-voice-profiles.static.hf.space/profile_emolia_c0542.html):
the **40 intense·free EmoNet emotions** plus **12 edge-case bursts** (fear/pain screams,
whimpering, sad crying, tears, pain groan, cold shiver, amused laughter).

---

## 1 · Batched multi-seed generation (the speed trick)

Chatterbox VC synthesizes the mel-spectrogram with a Conditional Flow-Matching (CFM) decoder.
The decoder is initialised from Gaussian noise:

```python
# chatterbox/models/s3gen/flow_matching.py
z = torch.randn_like(mu) * temperature   # <- the only source of run-to-run variation
```

To get **N candidates** we do **not** loop N times. We tokenize the source **once** and repeat
the content tokens along the batch dimension, so a single forward pass draws N independent
noises `(N, 80, T)` and returns N distinct conversions:

```python
# scripts/vc_batch.py  ::  convert_n()
s3_tokens, _ = s3gen.tokenizer(audio_16)      # (1, T)
tok = s3_tokens.repeat(n, 1).contiguous()      # (N, T)  -> N candidates in ONE pass
wavs, _ = s3gen.inference(speech_tokens=tok, ref_dict=vc.ref_dict)   # (N, samples)
```

**Why it matters.** Measured on one A100 (24 kHz output):

| batch N | ms / candidate | RTF (gen-s per s of audio) | speedup vs N=1 |
|--:|--:|--:|--:|
| 1  | ~530 | 0.19  | 1.0× |
| 8  | ~210 | 0.058 | ~2.5× |
| 32 | ~180 | 0.037 | ~2.4–3.8× |

Batching amortises the model's fixed per-call overhead, so 32 candidates cost far less than
32× a single call. The target speaker embedding is computed **once** and cached in `ref_dict`.

**Determinism.** Each pass seeds `torch.manual_seed(seed)` with `seed = 1234 + source_index`.
Every later stage (SIDON, speaker-sim, reward) regenerates **the exact same 32 candidates**, so
all scores are index-aligned and the whole study is reproducible.

---

## 2 · Scoring — emotion, quality, speaker similarity

All experts read a **frozen BUD-E-Whisper** encoder embedding `[1, 1500, 768]`, so we encode
each candidate once and fan out to every expert (batched over the 32 candidates → ~9 ms/candidate).

- **Emotion** (`laion/Empathic-Insight-Voice-Small`): 40 full-sequence MLP experts, one per
  EmoNet emotion, each returning an intensity in `[0, ~4.5]`.
  - **emotion cosine** = cosine similarity between the source's top-3 emotion vector and the
    candidate's values on those same 3 dims → *did we keep the same emotional direction?*
  - **emotion MAE** = mean absolute error on those 3 dims → *did we keep the same intensity?*
  - **peak emotion** = `max` over all 40 experts → *how strongly is any emotion expressed?*
    (used by the reward, §4).
- **Quality** (`laion/Empathic-Insight-Voice-Plus`): 4 pooled-feature MLPs distilled from DNSMOS
  and AudioBox. We use **Overall-Quality** and **Speech-Quality** (MOS-like). Pooling is
  `concat(mean, min, max, std)` over the sequence → `[1, 3072]`.
- **Speaker similarity to the target**, with two independent embeddings:
  - **ECAPA-TDNN** (192-d, `speechbrain/spkrec-ecapa-voxceleb`) — prosody-sensitive x-vector.
  - **Orange Speaker-wavLM-tbr** (128-d, L2-normalised) — a **timbre**-focused embedding.
  - Both reported as cosine-to-target. Two embeddings give a second opinion because they
    disagree about how much identity a step (e.g. SIDON) actually moves.

> **Note on speaker-sim interpretation.** In this experiment the sources and the target are the
> *same nominal speaker* (`emolia_c0542`) performing different emotions, so speaker-sim here
> measures **identity consistency**, not cross-speaker conversion accuracy. Source→target
> baseline: ECAPA **0.47**, Orange **0.69**. Interestingly, voice conversion **raises** the
> ECAPA match above the emotional source (→ ~0.64): it re-timbres from the clean reference.

---

## 3 · SIDON post-processing

Each candidate is passed through [SIDON](https://github.com/sarulab-speech/Sidon)
(`sarulab-speech/sidon-v0.1`), a speech-restoration model: a w2v-BERT 2.0 feature extractor →
a TorchScript feature encoder → a TorchScript decoder that resynthesises a clean **48 kHz**
waveform. Implementation mirrors the reference recipe (high-pass at 50 Hz, 16 kHz features,
chunked with a 1-frame cache):

```python
# scripts/refine_pass.py :: sidon_restore()
wav_16k = torchaudio.functional.highpass_biquad(w, sr, 50)
wav_16k = torchaudio.functional.resample(wav_16k, sr, 16000)
for chunk in wav_16k.view(-1).split(16000*96):
    feature = fe(pre(chunk)["input_features"])["last_hidden_state"]
    restoreds.append(decoder(feature.transpose(1,2)).view(-1)[:-960])
```

**Measured effect** (mean over all 32 × 52 = 1 664 candidates):

| metric | raw Chatterbox | + SIDON | Δ |
|---|--:|--:|--:|
| Overall-Q | 2.99 | 3.02 | **+0.03** |
| Speech-Q  | 1.96 | 1.99 | **+0.04** |
| emotion-MAE | 0.215 | 0.231 | +0.016 |
| emotion-cos | 0.993 | 0.984 | −0.008 |
| ECAPA speaker-sim | 0.643 | 0.595 | **−0.048** |
| Orange timbre-sim | 0.762 | 0.763 | +0.001 |

**Reading.** On these already-clean 24 kHz conversions, SIDON buys a small quality bump and
holds emotion, but **costs ECAPA speaker similarity** — the restoration decoder resynthesises
timbre, which the prosody-sensitive ECAPA notices while the timbre-focused Orange embedding does
not. So SIDON is a quality/identity trade, not a free lunch.

---

## 4 · The reward — expressive **and** clean

Best-by-quality and best-by-emotion are single-objective. The **reward** combines both so a
candidate must be *expressive and clean at the same time*. For every candidate:

```
reward = z_pool(peak_emotion) + z_pool(Overall_Quality)
```

where `peak_emotion = max over the 40 emotion experts`, and `z_pool` is z-normalisation over the
**entire pool** (all 1 664 candidates), computed separately within the raw pool and within the
SIDON pool. Summing two z-scores weights expressivity and quality equally on comparable scales.

The `⭐` pick in each demo card is the reward-maximising candidate. The **global leaderboard** is
topped by the edge-case screams (peak emotion ≈ 3.6, *Distress*, at Overall-Q ≈ 3.2) — the takes
that commit hardest to the emotion without going noisy.

---

## 5 · Diminishing returns — how many candidates?

Expected best-of-k **reward** (after SIDON), averaged over the 52 groups, as a fraction of the
full 1→32 gain:

| k | reward | % of gain | Δ vs previous |
|--:|--:|--:|--:|
| 1  | 0.000 | 0%  | — |
| 4  | 0.277 | 51% | +0.277 |
| 8  | 0.382 | 71% | +0.105 |
| 16 | 0.469 | 87% | +0.087 |
| 32 | 0.541 | 100%| +0.072 |

This reward deliberately chases the **most intense** take, so unlike pure quality — which
saturates by k≈8 — it keeps climbing slowly into the tail (you are sampling an extreme-value
statistic). There is **no sharp knee**; the decay is gradual (+0.10 → +0.09 → +0.07 per
doubling). Practical guidance:

- **k = 4** already captures ~half the gain — cheapest useful setting.
- **⭐ k = 8 is the sweet spot (recommended default).** Quality is fully saturated, ~70% of the
  expressive gain is captured, at ¼ the cost of best-of-32.
- **k = 16–32** only pays off if you are specifically mining the most extreme expressive takes;
  cost is linear in k, so 16→32 doubles compute for the last ~13%.

---

## 6 · Cost — GPU-hours for 1,000,000 samples

Per-candidate wall-time on one A100 (5.1 s reference clip):

| stage | time / candidate |
|---|--:|
| generation (batched) | 182 ms |
| **SIDON restoration** | **255 ms** ← dominant |
| scoring (40 emotion + 4 quality experts, batched) | 9 ms |

Because SIDON costs more than generation, the **efficient production design ranks first and
runs SIDON only on the winner** (×1, not ×N). GPU-hours to process 1,000,000 source clips:

| N candidates | efficient (rank → SIDON winner) | SIDON-all-then-rank |
|--:|--:|--:|
| 4  | 284 GPU-h | 496 GPU-h |
| **8** | **496 GPU-h** | 993 GPU-h |
| 16 | 922 GPU-h | 1 986 GPU-h |
| 32 | 1 772 GPU-h | 3 972 GPU-h |

At the recommended **N=8**, that is **≈496 GPU-hours per 1M** (~2.6 days on 8×A100) with the
efficient pipeline — refining only the winner roughly **halves** the bill versus SIDON-on-all.
Scoring every emotion+quality expert is nearly free. The estimate is compute-only at 100%
utilisation; real runs add ~20–30% for I/O and MP3 encoding and scale ~linearly with clip length.

---

## 7 · Reproduce it

See [`README.md`](./README.md) for the exact run order and commands. In short:

1. `vc_batch.py` — batched multi-seed VC core (`convert_n`).
2. `worker.py` — generate 32 candidates/source, score emotion top-3 + quality, save top-3 by a
   blended pick. (8-GPU shardable.)
3. `refine_pass.py` — regenerate the same 32, add ECAPA+Orange speaker-sim, SIDON-refine and
   re-score; save re-ranked raw/SIDON picks (by quality and by emotion-MAE).
4. `reward_pass.py` — score all 40 emotions → peak intensity; cache every candidate MP3.
5. `src_spk.py` — source→target speaker-sim baseline.
6. `bench_stages.py`, `rtf_sweep.py` — timing.
7. `build_report2.py`, `build_report3.py` — best-of-k, SIDON effect, reward, diminishing
   returns, GPU-hours (writes the JSON in [`results/`](./results)).
8. `make_html3.py` — build the demo page.

**Live demo:** https://tts-agi-chatterbox-expressive-bestofn.static.hf.space
