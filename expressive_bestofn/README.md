# Expressive Best-of-N Voice Conversion

**Make Chatterbox voice conversion sound as good as possible while keeping the emotion of the
original performance — by generating many candidates, scoring them, and keeping the best.**

Voice conversion is stochastic (the flow-matching decoder starts from Gaussian noise), so any
single run is a lucky or unlucky draw. This pipeline instead **samples N candidates in one
batched pass**, scores each for **emotion fidelity**, **audio quality**, and **speaker/timbre
similarity**, optionally **restores** them with **SIDON**, and **ranks** them with a reward that
rewards takes that are *expressive and clean at the same time*.

### ⭐ Sweet spot: **best-of-8**

Generate **8 candidates** per source. That is the recommended operating point: audio quality is
already fully saturated at N≈8, it captures **~70%** of the total expressivity gain (out of the
full 1→32 range), and it costs **¼ of best-of-32**. Going to 16 or 32 only pays off if you are
specifically mining the most extreme expressive takes — the returns past 8 are a thin, linear-cost
tail. For maximum efficiency, **rank the 8 candidates first and run SIDON only on the winner**
(≈496 GPU-hours per 1,000,000 samples; roughly half the cost of restoring all 8).

### ▶️ Live demo
**https://tts-agi-chatterbox-expressive-bestofn.static.hf.space**

52 sources (40 intense·free EmoNet emotions + 12 edge-case bursts) converted to the voice
*"Measured Slavic Historian"* (`emolia_c0542`), each with its reward pick, best-by-quality and
best-by-emotion-MAE takes, before/after SIDON, plus best-of-k curves, the SIDON effect, the
diminishing-returns analysis and the 1M-sample GPU-hours table.

📄 **Full method write-up:** [`METHOD.md`](./METHOD.md) — what is done, why, and how, in detail.

---

## Why

- **Quality varies seed-to-seed** → sample several, keep the best.
- **Naively optimising quality can flatten emotion** → score emotion explicitly and rank on a
  reward that balances expressivity against quality, so the winner is both.
- **Batching is nearly free** → 32 candidates in one forward pass cost ~3–4× *less per candidate*
  than looping (fixed per-call overhead is amortised).

## How (one glance)

```
source clip ──► Chatterbox VC (batched N seeds, one pass) ──► N candidates
                                     │
             ┌───────────────────────┼────────────────────────────┐
             ▼                       ▼                             ▼
   emotion experts (×40)     quality experts (×4)        speaker-sim (ECAPA + Orange)
   Empathic-Insight-Small    Empathic-Insight-Plus       identity / timbre to target
             │                       │
             └────────► reward = z(peak-emotion) + z(Overall-Q) ──► rank
                                     │
                          (optional) SIDON restoration on the winner ──► final 48 kHz take
```

## Models used

| Model | Link |
|---|---|
| Empathic-Insight-Voice-Plus (quality experts) | https://huggingface.co/laion/Empathic-Insight-Voice-Plus |
| Empathic-Insight-Voice-Small (40 emotion experts) | https://huggingface.co/laion/Empathic-Insight-Voice-Small |
| BUD-E-Whisper (frozen encoder) | https://huggingface.co/laion/BUD-E-Whisper |
| SIDON speech restoration | https://github.com/sarulab-speech/Sidon · https://huggingface.co/sarulab-speech/sidon-v0.1 |
| ECAPA-TDNN speaker embedding | https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb |
| Orange Speaker-wavLM-tbr timbre embedding | https://huggingface.co/Orange/Speaker-wavLM-tbr |

## Key results (this run, 52 sources × 32 candidates)

- **Throughput:** batched RTF ~0.037, ~180 ms/candidate at N=32 (vs ~530 ms at N=1).
- **Best-of-N buys clean audio at equal emotion:** emotion cosine is already ~0.99 at N=1;
  1→32 mostly halves emotion-MAE (0.215→0.125) and adds ~0.05 MOS.
- **SIDON:** +0.03 Overall-Q / +0.04 Speech-Q, holds emotion, but −0.048 ECAPA speaker-sim
  (timbre resynthesis); Orange timbre-sim unchanged.
- **Diminishing returns:** gradual, no sharp knee — k=4 → 51%, **k=8 → 71%**, k=16 → 87% of the
  1→32 reward gain. Recommended operating point **N=8**.
- **Cost:** **≈496 GPU-hours per 1,000,000 samples** at N=8 with the efficient
  *rank-then-SIDON-the-winner* design (~half of SIDON-on-all).

---

## Reproduce

### Dependencies
```bash
pip install -e ..                       # the chatterbox_vc / chatterbox-tts library (parent repo)
pip install speechbrain hyperpyyaml     # ECAPA + Orange embedding loader
pip install transformers librosa soundfile torchaudio
```
The Orange embedding loader is `spk_embeddings.py` (WavLM-based `EmbeddingsModel`); point
`sys.path` at a directory that contains it (see `refine_pass.py` / `src_spk.py`). Scoring needs
the three LAION model repos above (auto-downloaded from the Hub).

### Working directory & env
The scripts use a scratch tree (default `/tmp/vcbon`, RAM-backed here) with:
`src/` (source clips), `target/reference.mp3` (target voice), `out/` (scores + audio + JSON).
Scorer model paths are passed via env:
```bash
export WHISPER_DIR=<snapshot of laion/BUD-E-Whisper>
export EMO_DIR=<snapshot of laion/Empathic-Insight-Voice-Small (the *.pth experts)>
export QUAL_DIR=<snapshot of laion/Empathic-Insight-Voice-Plus (the *.pth experts)>
```
Each generation seeds `1234 + source_index`, so every stage regenerates the **same** 32
candidates and all scores stay index-aligned.

### Run order (each `*_pass.py` is 8-GPU shardable: `python x.py <shard> <n_shards>`)
```bash
# 1. generate 32 candidates/source, score emotion top-3 + quality, save top-3
CUDA_VISIBLE_DEVICES=0 python scripts/worker.py 0 8      # ... shards 0..7

# 2. speaker-sim (ECAPA + Orange) + SIDON refine + re-score, save re-ranked picks
CUDA_VISIBLE_DEVICES=0 python scripts/refine_pass.py 0 8

# 3. peak-emotion (all 40) + cache every candidate mp3 for reward re-ranking
CUDA_VISIBLE_DEVICES=0 python scripts/reward_pass.py 0 8

# 4. source→target speaker-sim baseline (single GPU)
CUDA_VISIBLE_DEVICES=0 python scripts/src_spk.py

# 5. timing (single GPU)
CUDA_VISIBLE_DEVICES=0 python scripts/bench_stages.py
CUDA_VISIBLE_DEVICES=0 python scripts/rtf_sweep.py

# 6. analysis -> results/*.json
python scripts/build_report2.py      # best-of-k (raw vs SIDON), SIDON effect, speaker-sim
python scripts/build_report3.py      # reward, diminishing returns, GPU-hours, leaderboard

# 7. build the demo page -> out/index.html
python scripts/make_html3.py
```

### What's in this folder
- [`METHOD.md`](./METHOD.md) — the detailed explanation.
- `scripts/` — the exact pipeline (original filenames preserve the import graph:
  `worker.py` imports `vc_batch`, `refine_pass.py`/`reward_pass.py` import `worker`+`vc_batch`).
- `results/` — the computed analysis JSON (see [`results/README.md`](./results/README.md)).
- `demo/index.html` — a reference copy of the demo page (audio is hosted on the live Space above).

> This is an add-on to the Chatterbox Voice Conversion library in the parent directory; it does
> not modify the core `chatterbox_vc` package.
