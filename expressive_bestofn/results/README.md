# Analysis artifacts

Computed by `scripts/build_report2.py` and `scripts/build_report3.py` from the per-candidate
scores. All means are over the 52 sources × 32 candidates unless noted.

| file | contents |
|---|---|
| `runstats.json` | live-run throughput: mean gen-seconds for a group of 32, mean batched RTF, total candidates |
| `rtf_sweep.json` | batch-size sweep (N=1/4/8/16/32) on short/median/long clips: gen time, ms/candidate, RTF, speedup vs N=1 |
| `stage_bench.json` | per-stage wall-time (generation / SIDON / scoring) for one 32-candidate group |
| `bestofk.json` | raw-only best-of-k (emotion cos/MAE, Overall-Q, Speech-Q, blended pick) |
| `bestofk2.json` | best-of-k for **raw vs SIDON**: emotion cos/MAE, Overall-Q, ECAPA, Orange |
| `sidon_effect.json` | mean metric before/after SIDON and the delta, per metric |
| `src_spk.json` | source→target speaker-sim baseline (ECAPA, Orange) per source |
| `reward_bestofk.json` | expected best-of-k **reward** (raw & SIDON) + marginal gain / % captured per k |
| `reward_picks.json` | per-source reward winner (index, reward, peak emotion + which, Overall-Q) for raw and SIDON |
| `reward_top60.json` | global reward leaderboard (top 60 candidates, SIDON pool) |
| `gpu_hours.json` | per-candidate stage times, 1M-sample GPU-hours (efficient vs SIDON-all) for N=4/8/16/32, recommended & 90%-knee k |

Definitions: **emotion cosine/MAE** compare source-vs-candidate on the source's top-3 emotions;
**peak emotion** = max over all 40 emotion experts; **reward** = `z_pool(peak_emotion) +
z_pool(Overall_Quality)` (z-normalised over the whole pool). See [`../METHOD.md`](../METHOD.md).
