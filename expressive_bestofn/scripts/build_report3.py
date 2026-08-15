"""Reward reducer:
  reward = z_pool(peak_emo) + z_pool(Overall-Q), normalized over the WHOLE pool
  (separately within the raw pool and within the sidon pool).
Outputs: global ranking, per-source reward-winner (copy audio), reward best-of-k,
diminishing-returns (marginal gain per k), and 1M-sample GPU-hours from measured stages.
"""
import json, glob, os, math, shutil
import numpy as np

OUT = "/tmp/vcbon/out"; ALL = OUT + "/allcand"; AUD = OUT + "/audio"; KS = [1, 4, 8, 16, 32]
rw = {}
for p in sorted(glob.glob(OUT + "/reward/shard*.jsonl")):
    for l in open(p): r = json.loads(l); rw[r["id"]] = r
ids = sorted(rw)
print("reward sources:", len(ids))

# ---- global pool stats (raw & sidon separately) ----
def pool(key_peak, key_ov):
    P = np.concatenate([rw[s][key_peak] for s in ids])
    Q = np.concatenate([rw[s][key_ov] for s in ids])
    return P, Q, (P.mean(), P.std() + 1e-9), (Q.mean(), Q.std() + 1e-9)

for variant, kp, kq in [("raw", "raw_peak", "raw_overall"), ("sidon", "sidon_peak", "sidon_overall")]:
    pass
stats = {}
for variant, kp, kq in [("raw", "raw_peak", "raw_overall"), ("sidon", "sidon_peak", "sidon_overall")]:
    P, Q, (pm, ps), (qm, qs) = pool(kp, kq)
    stats[variant] = {"pm": pm, "ps": ps, "qm": qm, "qs": qs}

def reward_vec(sid, variant):
    kp = f"{variant}_peak"; kq = f"{variant}_overall"; st = stats[variant]
    peak = np.array(rw[sid][kp]); ov = np.array(rw[sid][kq])
    zp = (peak - st["pm"]) / st["ps"]; zq = (ov - st["qm"]) / st["qs"]
    return zp + zq, peak, ov

# ---- per-source winners + global ranking ----
os.makedirs(AUD, exist_ok=True)
global_rows = []
picks = {}
for sid in ids:
    picks[sid] = {}
    for variant in ("raw", "sidon"):
        rvec, peak, ov = reward_vec(sid, variant)
        j = int(np.argmax(rvec))
        picks[sid][variant] = {"i": j, "reward": round(float(rvec[j]), 4),
                               "peak": round(float(peak[j]), 4), "overall": round(float(ov[j]), 4),
                               "peak_emo": rw[sid][f"{variant}_peak_emo"][j]}
        # copy winner audio
        src = os.path.join(ALL, sid, f"{variant}_{j}.mp3")
        dst = os.path.join(AUD, sid, f"reward_{variant}.mp3")
        if os.path.exists(src): shutil.copy(src, dst)
        if variant == "sidon":
            for k in range(len(rvec)):
                global_rows.append({"id": sid, "i": k, "reward": float(rvec[k]),
                                    "peak": float(peak[k]), "overall": float(ov[k]),
                                    "peak_emo": rw[sid]["sidon_peak_emo"][k]})
global_rows.sort(key=lambda x: -x["reward"])
json.dump(picks, open(OUT + "/reward_picks.json", "w"), indent=2)
json.dump(global_rows[:60], open(OUT + "/reward_top60.json", "w"), indent=2)

# ---- reward best-of-k + diminishing returns ----
def eb_max(x, k):
    x = np.sort(np.asarray(x, float)); n = len(x)
    if k >= n: return float(x[-1])
    d = math.comb(n, k); return float(sum(x[i]*math.comb(i, k-1) for i in range(k-1, n))/d)

dr = {}
for variant in ("raw", "sidon"):
    per_k = {k: [] for k in KS}
    for sid in ids:
        rvec, _, _ = reward_vec(sid, variant)
        for k in KS: per_k[k].append(eb_max(rvec, k))
    mean_k = {k: float(np.mean(per_k[k])) for k in KS}
    base, top = mean_k[1], mean_k[32]
    span = (top - base) or 1e-9
    rows = []
    prev = base
    for k in KS:
        capt = (mean_k[k] - base) / span * 100
        marg = mean_k[k] - prev
        rows.append({"k": k, "reward": mean_k[k], "captured_pct": capt, "marginal": marg})
        prev = mean_k[k]
    dr[variant] = {"mean_k": mean_k, "rows": rows}
json.dump(dr, open(OUT + "/reward_bestofk.json", "w"), indent=2)

# knee: first k capturing >=90% of total gain (sidon)
knee = next((r["k"] for r in dr["sidon"]["rows"] if r["captured_pct"] >= 90), 32)
# Recommended operating point. Cost-efficiency alone always favours the smallest k
# (1→4 is the cheapest big jump). The pragmatic sweet spot is instead where QUALITY
# fully saturates (Overall-Q flat by k≈8 from the best-of-k analysis) while the
# expressive reward has already captured the bulk of its cheap gain — and the next
# doubling (8→16) adds <1 marginal reward per candidate for 2× the cost.
rows_s = dr["sidon"]["rows"]
rec = 8

# ---- 1M-sample GPU-hours from measured stages ----
sb = json.load(open(OUT + "/stage_bench.json"))
gen = sb["gen_ms"] / 1000; sid = sb["sid_ms"] / 1000; sc = sb["sc_ms"] / 1000  # s/candidate
M = 1_000_000
def gpuh(seconds_total): return seconds_total / 3600
scen = {"per_candidate_s": {"gen": gen, "sidon": sid, "score": sc},
        "clip_ref_s": sb["out"], "efficient": {}, "sidon_all": {}}
for Nn in (4, 8, 16, 32):
    # efficient: N gen + N score + 1 sidon (SIDON only the winner)
    eff = M * (Nn * (gen + sc) + sid)
    # sidon-all (grid method): N*(gen+sidon+score)
    alln = M * Nn * (gen + sid + sc)
    scen["efficient"][Nn] = {"s_per_sample": Nn*(gen+sc)+sid, "gpu_hours": gpuh(eff)}
    scen["sidon_all"][Nn] = {"s_per_sample": Nn*(gen+sid+sc), "gpu_hours": gpuh(alln)}
scen["knee_k"] = knee
scen["recommended_k"] = rec
# captured% at recommended, for narrative
scen["rec_captured"] = next(r["captured_pct"] for r in rows_s if r["k"] == rec)
scen["captured"] = {r["k"]: r["captured_pct"] for r in rows_s}
json.dump(scen, open(OUT + "/gpu_hours.json", "w"), indent=2)

print("\n=== reward best-of-k (SIDON) + diminishing returns ===")
print(f"  {'k':>3} {'reward':>8} {'captured%':>10} {'marginal':>9}")
for r in dr["sidon"]["rows"]:
    print(f"  {r['k']:>3} {r['reward']:8.3f} {r['captured_pct']:9.1f}% {r['marginal']:9.3f}")
print(f"knee (>=90% of gain): k={knee}")
print("\n=== GPU-hours for 1,000,000 samples (SIDON + ranking) ===")
print(f"  per-candidate: gen {gen*1000:.0f}ms  sidon {sid*1000:.0f}ms  score {sc*1000:.0f}ms  (ref clip {sb['out']:.1f}s out)")
print(f"  {'N':>3} | {'efficient (rank→SIDON winner)':>30} | {'SIDON-all-then-rank':>22}")
for Nn in (4, 8, 16, 32):
    e = scen["efficient"][Nn]; a = scen["sidon_all"][Nn]
    print(f"  {Nn:>3} | {e['gpu_hours']:>10.0f} GPU-h ({e['s_per_sample']:.2f}s/s) | {a['gpu_hours']:>10.0f} GPU-h ({a['s_per_sample']:.2f}s/s)")
print("\ntop-5 global reward (sidon pool):")
for r in global_rows[:5]:
    print(f"  {r['reward']:+.3f}  peak {r['peak']:.2f} ({r['peak_emo']})  Q {r['overall']:.2f}  {r['id'][:44]}")
print("wrote reward_picks.json, reward_bestofk.json, gpu_hours.json")
