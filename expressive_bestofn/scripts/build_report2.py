"""Merge RAW (worker) + SIDON/speaker (refine) -> best-of-k (raw vs sidon) for
emo-cos/emo-MAE/Overall-Q/Speech-Q/ECAPA/Orange, SIDON-effect deltas, and the grid
(re-ranked raw & SIDON picks by quality and by emo-MAE, with speaker-sim badges)."""
import json, glob, os, math, html
import numpy as np

OUT = "/tmp/vcbon/out"
KS = [1, 4, 8, 16, 32]

raw = {}
for p in sorted(glob.glob(OUT + "/scores/shard*.jsonl")):
    for l in open(p):
        r = json.loads(l); raw[r["id"]] = r
ref = {}
for p in sorted(glob.glob(OUT + "/refine/shard*.jsonl")):
    for l in open(p):
        r = json.loads(l); ref[r["id"]] = r
ids = sorted(set(raw) & set(ref))
src_spk = json.load(open(OUT + "/src_spk.json")) if os.path.exists(OUT + "/src_spk.json") else {}
bestofk = json.load(open(OUT + "/bestofk.json"))
runstats = json.load(open(OUT + "/runstats.json"))
rtf = json.load(open(OUT + "/rtf_sweep.json")) if os.path.exists(OUT + "/rtf_sweep.json") else None
print("merged sources:", len(ids))


def eb_max(x, k):
    x = np.sort(np.asarray(x, float)); n = len(x)
    if k >= n: return float(x[-1])
    d = math.comb(n, k); return float(sum(x[i]*math.comb(i, k-1) for i in range(k-1, n))/d)

def eb_min(x, k):
    x = np.sort(np.asarray(x, float)); n = len(x)
    if k >= n: return float(x[0])
    d = math.comb(n, k); return float(sum(x[i]*math.comb(n-1-i, k-1) for i in range(0, n-k+1))/d)


# ---- per-source metric vectors (index-aligned, 32 candidates) ----
def vecs(sid):
    rc = raw[sid]["cands"]; sd = ref[sid]["sidon"]
    return {
      "raw": {"cos":[c["emo_cos"] for c in rc], "mae":[c["emo_mae"] for c in rc],
              "ov":[c["quality"]["Overall_Quality"] for c in rc], "sp":[c["quality"]["Speech_Quality"] for c in rc],
              "ec":ref[sid]["raw_ecapa"], "or":ref[sid]["raw_orange"]},
      "sidon": {"cos":[c["emo_cos"] for c in sd], "mae":[c["emo_mae"] for c in sd],
                "ov":[c["quality"]["Overall_Quality"] for c in sd], "sp":[c["quality"]["Speech_Quality"] for c in sd],
                "ec":[c["ecapa"] for c in sd], "or":[c["orange"] for c in sd]},
    }

# best-of-k averaged across sources, for raw and sidon
curve = {v: {k: {m: [] for m in ("cos","mae","ov","sp","ec","or")} for k in KS} for v in ("raw","sidon")}
for sid in ids:
    V = vecs(sid)
    for v in ("raw","sidon"):
        for k in KS:
            curve[v][k]["cos"].append(eb_max(V[v]["cos"], k))
            curve[v][k]["mae"].append(eb_min(V[v]["mae"], k))
            curve[v][k]["ov"].append(eb_max(V[v]["ov"], k))
            curve[v][k]["sp"].append(eb_max(V[v]["sp"], k))
            curve[v][k]["ec"].append(eb_max(V[v]["ec"], k))
            curve[v][k]["or"].append(eb_max(V[v]["or"], k))
curve_mean = {v: {k: {m: float(np.mean(x)) for m, x in d.items()} for k, d in kv.items()} for v, kv in curve.items()}
json.dump(curve_mean, open(OUT + "/bestofk2.json", "w"), indent=2)

# ---- SIDON effect: per-candidate delta means (sidon - raw), aligned by index ----
eff = {m: {"raw": [], "sidon": []} for m in ("cos","mae","ov","sp","ec","or")}
for sid in ids:
    V = vecs(sid)
    for m in eff:
        eff[m]["raw"] += V["raw"][m]; eff[m]["sidon"] += V["sidon"][m]
sidon_effect = {}
for m in eff:
    r = np.array(eff[m]["raw"]); s = np.array(eff[m]["sidon"])
    sidon_effect[m] = {"raw_mean": float(r.mean()), "sidon_mean": float(s.mean()),
                       "delta": float((s-r).mean()), "delta_std": float((s-r).std())}
json.dump(sidon_effect, open(OUT + "/sidon_effect.json", "w"), indent=2)

print("\nBest-of-k (raw vs sidon), mean across sources:")
print(f"  {'k':>3} | {'RAW cos':>7} {'mae':>6} {'ovQ':>6} {'ECAPA':>6} {'Oran':>6} | {'SID cos':>7} {'mae':>6} {'ovQ':>6} {'ECAPA':>6} {'Oran':>6}")
for k in KS:
    a = curve_mean["raw"][k]; b = curve_mean["sidon"][k]
    print(f"  {k:>3} | {a['cos']:7.4f} {a['mae']:6.3f} {a['ov']:6.3f} {a['ec']:6.3f} {a['or']:6.3f} | "
          f"{b['cos']:7.4f} {b['mae']:6.3f} {b['ov']:6.3f} {b['ec']:6.3f} {b['or']:6.3f}")
print("\nSIDON effect (mean over all 32x{} candidates):".format(len(ids)))
names = {"cos":"emo-cos","mae":"emo-MAE","ov":"Overall-Q","sp":"Speech-Q","ec":"ECAPA-sim","or":"Orange-sim"}
for m in ("cos","mae","ov","sp","ec","or"):
    e = sidon_effect[m]
    print(f"  {names[m]:10s} raw {e['raw_mean']:.3f} -> sidon {e['sidon_mean']:.3f}  (Δ {e['delta']:+.3f})")
if src_spk:
    se = np.mean([src_spk[s]["ecapa"] for s in src_spk]); so = np.mean([src_spk[s]["orange"] for s in src_spk])
    print(f"\nSource→target identity baseline: ECAPA {se:.3f}  Orange {so:.3f}")
print("wrote bestofk2.json, sidon_effect.json")
