"""v3 grid: throughput, best-of-k (raw vs SIDON), speaker-sim, SIDON effect,
+ EXPRESSIVITY×QUALITY REWARD (peak-emotion z + Overall-Q z), reward best-of-k /
diminishing returns, global reward leaderboard, and 1M-sample GPU-hours."""
import json, glob, os, html
import numpy as np
OUT = "/tmp/vcbon/out"; KS = [1, 4, 8, 16, 32]
raw = {}
for p in sorted(glob.glob(OUT+"/scores/shard*.jsonl")):
    for l in open(p): r = json.loads(l); raw[r["id"]] = r
ref = {}
for p in sorted(glob.glob(OUT+"/refine/shard*.jsonl")):
    for l in open(p): r = json.loads(l); ref[r["id"]] = r
ids = sorted(set(raw) & set(ref))
bk = json.load(open(OUT+"/bestofk2.json")); eff = json.load(open(OUT+"/sidon_effect.json"))
runstats = json.load(open(OUT+"/runstats.json")); rtf = json.load(open(OUT+"/rtf_sweep.json"))
src_spk = json.load(open(OUT+"/src_spk.json")); picks = json.load(open(OUT+"/reward_picks.json"))
rbk = json.load(open(OUT+"/reward_bestofk.json")); gph = json.load(open(OUT+"/gpu_hours.json"))
top60 = json.load(open(OUT+"/reward_top60.json"))

def esc(s): return html.escape(str(s))
def label_of(sid):
    p = sid.split("__")
    if "__E__" in sid: return p[-3].replace("_"," "), "emotion"
    if "__X__" in sid: return p[-2].replace("_"," "), "edge"
    return sid, "other"
def chips(d, scale=4.0):
    return "".join(f"<span class=chip><b>{esc(k.replace('_',' '))}</b> {v:.2f}<span class=bar style='width:{max(6,min(100,v/scale*100)):.0f}%'></span></span>" for k,v in sorted(d.items(),key=lambda x:-x[1]))
def bdg(l,v,cls=""): return f"<span class='badge {cls}'>{esc(l)} <b>{v}</b></span>"

def sc_from(sid, variant, kind):
    """kind: bestQ|bestMAE from refine picks; player badges."""
    return ref[sid]["picks"][f"{variant}_{kind}"]
def player(sid, fname, tag, sc, star=False, extra=""):
    b=(bdg("emo-cos",f"{sc['emo_cos']:.3f}","g")+bdg("emo-MAE",f"{sc['emo_mae']:.2f}")
       +bdg("Ovr-Q",f"{sc['Overall_Quality']:.2f}","b")+bdg("Spch-Q",f"{sc['Speech_Quality']:.2f}")
       +bdg("ECAPA",f"{sc['ecapa']:.2f}","p")+bdg("Orange",f"{sc['orange']:.2f}","o")+extra)
    return (f"<div class='cell{' best' if star else ''}'><div class=ctag>{esc(tag)}</div>"
            f"<audio controls preload=none src='audio/{esc(sid)}/{fname}'></audio><div class=meta>{b}</div></div>")

def reward_player(sid, variant):
    pk = picks[sid][variant]
    b=(bdg("REWARD",f"{pk['reward']:+.2f}","r")+bdg("peak-emo",f"{pk['peak']:.2f}","g")
       +f"<span class=chip style='background:#2a2233'>{esc(pk['peak_emo'].replace('_',' '))}</span>"
       +bdg("Ovr-Q",f"{pk['overall']:.2f}","b"))
    tag = "REWARD pick · Chatterbox" if variant=="raw" else "REWARD pick · +SIDON ★"
    return (f"<div class='cell{' rew' if variant=='sidon' else ''}'><div class=ctag>{tag}</div>"
            f"<audio controls preload=none src='audio/{esc(sid)}/reward_{variant}.mp3'></audio><div class=meta>{b}</div></div>")

def card(sid):
    lbl,kind = label_of(sid); r=raw[sid]
    src_top3=dict(zip(r["top3_emotions"],r["src_top3"]))
    sb = f" · src→target ECAPA {src_spk[sid]['ecapa']:.2f} · Orange {src_spk[sid]['orange']:.2f}" if sid in src_spk else ""
    h=[f"<div class='card {kind}'>"]
    h.append(f"<div class=hdr><span class=lbl>{esc(lbl)}</span><span class=kind>{'INTENSE · FREE' if kind=='emotion' else 'EDGE CASE'}</span></div>")
    h.append("<div class=srcemo>source top-3 emotion read: "+chips(src_top3)+"</div>")
    h.append(f"<div class=row><div class=cell><div class=ctag>SOURCE (original emotion clip)</div>"
             f"<audio controls preload=none src='audio/{esc(sid)}/source.mp3'></audio>"
             f"<div class=meta>{bdg('Ovr-Q',f'{r['src_quality']['Overall_Quality']:.2f}','b')}{bdg('Spch-Q',f'{r['src_quality']['Speech_Quality']:.2f}')}<span class=note>{sb}</span></div></div></div>")
    # REWARD row (headline)
    h.append("<div class=sub2>⭐ <b>Expressivity×Quality reward</b> pick — z(peak-emotion)+z(Overall-Q)</div><div class=row2>")
    h.append(reward_player(sid,"raw")); h.append(reward_player(sid,"sidon")); h.append("</div>")
    # quality / mae rows
    h.append("<div class=sub2>Best by <b>Quality</b> — Chatterbox vs. +SIDON</div><div class=row2>")
    h.append(player(sid,"raw_bestQ.mp3","Chatterbox VC",sc_from(sid,"raw","bestQ")))
    h.append(player(sid,"sidon_bestQ.mp3","+ SIDON",sc_from(sid,"sidon","bestQ"))); h.append("</div>")
    h.append("<div class=sub2>Best by <b>emotion-MAE</b> — Chatterbox vs. +SIDON</div><div class=row2>")
    h.append(player(sid,"raw_bestMAE.mp3","Chatterbox VC",sc_from(sid,"raw","bestMAE")))
    h.append(player(sid,"sidon_bestMAE.mp3","+ SIDON",sc_from(sid,"sidon","bestMAE"))); h.append("</div></div>")
    return "".join(h)

emo_cards="".join(card(s) for s in ids if label_of(s)[1]=="emotion")
edge_cards="".join(card(s) for s in ids if label_of(s)[1]=="edge")

# reward best-of-k / diminishing returns table
def dr_rows():
    out=[]
    for r in rbk["sidon"]["rows"]:
        out.append(f"<tr><td><b>{r['k']}</b></td><td>{r['reward']:.3f}</td><td>{r['captured_pct']:.1f}%</td>"
                   f"<td>{r['marginal']:+.3f}</td></tr>")
    return "".join(out)

def svg_dr():
    rows=rbk["sidon"]["rows"]; ys=[r["captured_pct"] for r in rows]
    W,H,pad=340,110,28
    pts=[]
    for j,r in enumerate(rows):
        x=pad+j*(W-2*pad)/(len(rows)-1); y=(H-pad)-(r["captured_pct"]/100)*(H-2*pad); pts.append((x,y))
    pl=" ".join(f"{x:.1f},{y:.1f}" for x,y in pts)
    dots="".join(f"<circle cx={x:.1f} cy={y:.1f} r=3 fill=#e8823a/>" for x,y in pts)
    lab="".join(f"<text x={pts[j][0]:.0f} y={H-8} font-size=10 text-anchor=middle fill=#888>{r['k']}</text>" for j,r in enumerate(rows))
    val="".join(f"<text x={pts[j][0]:.0f} y={pts[j][1]-7:.0f} font-size=9 text-anchor=middle fill=#e7ecf3>{r['captured_pct']:.0f}%</text>" for j,r in enumerate(rows))
    y90=(H-pad)-0.9*(H-2*pad)
    return (f"<svg viewBox='0 0 {W} {H}' width=100%><line x1={pad} y1={y90:.0f} x2={W-pad} y2={y90:.0f} stroke=#3aa76d stroke-dasharray=4 stroke-width=1/>"
            f"<text x={W-pad} y={y90-4:.0f} font-size=9 fill=#3aa76d text-anchor=end>90% of gain</text>"
            f"<polyline points='{pl}' fill=none stroke=#e8823a stroke-width=2/>{dots}{lab}{val}</svg>")

def top_rows():
    out=[]
    for i,r in enumerate(top60[:15]):
        lbl,_=label_of(r["id"])
        out.append(f"<tr><td>{i+1}</td><td>{esc(lbl)}</td><td>{r['reward']:+.3f}</td>"
                   f"<td>{r['peak']:.2f}</td><td>{esc(r['peak_emo'].replace('_',' '))}</td><td>{r['overall']:.2f}</td></tr>")
    return "".join(out)

def gph_rows():
    out=[]
    for N in (4,8,16,32):
        e=gph["efficient"][str(N)]; a=gph["sidon_all"][str(N)]
        star=" ★" if N==gph["recommended_k"] else ""
        out.append(f"<tr><td><b>{N}{star}</b></td><td>{e['s_per_sample']:.2f}s</td><td>{e['gpu_hours']:,.0f}</td>"
                   f"<td class=sep>{a['s_per_sample']:.2f}s</td><td>{a['gpu_hours']:,.0f}</td></tr>")
    return "".join(out)

# reuse charts + tables from v2 sections (best-of-k raw vs sidon, sidon effect, rtf)
def bk_table():
    rows=[]
    for k in KS:
        a=bk["raw"][str(k)]; b=bk["sidon"][str(k)]
        rows.append(f"<tr><td><b>{k}</b></td><td>{a['cos']:.4f}</td><td>{a['mae']:.3f}</td><td>{a['ov']:.3f}</td><td>{a['ec']:.3f}</td><td>{a['or']:.3f}</td>"
                    f"<td class=sep>{b['cos']:.4f}</td><td>{b['mae']:.3f}</td><td>{b['ov']:.3f}</td><td>{b['ec']:.3f}</td><td>{b['or']:.3f}</td></tr>")
    return "".join(rows)
names={"cos":"emotion cosine ↑","mae":"emotion MAE ↓","ov":"Overall quality ↑","sp":"Speech quality ↑","ec":"ECAPA speaker-sim ↑","or":"Orange timbre-sim ↑"}
def eff_rows():
    out=[]
    for m in ("cos","mae","ov","sp","ec","or"):
        e=eff[m]; d=e["delta"]; good=(d<0) if m=="mae" else (d>0)
        out.append(f"<tr><td>{names[m]}</td><td>{e['raw_mean']:.3f}</td><td>{e['sidon_mean']:.3f}</td><td class={'pos' if good else 'neg'}>{d:+.3f}</td></tr>")
    return "".join(out)
rr=[f"<tr><td>{x['clip']}</td><td>{x['src_s']:.1f}s</td><td>{x['B']}</td><td>{x['gen_s']:.2f}</td><td>{x['per_cand_ms']:.0f}</td><td>{x['rtf']:.3f}</td><td>{x['speedup']:.1f}×</td></tr>" for x in rtf["rows"]]
rtf_html="<table class=tbl><thead><tr><th>clip</th><th>src</th><th>B</th><th>gen s</th><th>ms/cand</th><th>RTF</th><th>speedup</th></tr></thead><tbody>"+"".join(rr)+"</tbody></table>"
se=np.mean([src_spk[s]["ecapa"] for s in src_spk]); so=np.mean([src_spk[s]["orange"] for s in src_spk])
knee=gph["knee_k"]; rec=gph["recommended_k"]; effK=gph["efficient"][str(rec)]; allK=gph["sidon_all"][str(rec)]
pc=gph["per_candidate_s"]; cap={int(k):v for k,v in gph["captured"].items()}

HTML=f"""<!doctype html><meta charset=utf-8><title>Chatterbox VC · reward ranking + SIDON</title>
<meta name=viewport content='width=device-width, initial-scale=1'>
<style>
:root{{--bg:#0e1116;--card:#171b22;--card2:#1e232c;--fg:#e7ecf3;--mut:#9aa7b8;--line:#2a3140;--acc:#5b8def;--sid:#e8823a;--rew:#d4b24a}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif}}
.wrap{{max-width:1200px;margin:0 auto;padding:28px 18px 80px}}
h1{{font-size:26px;margin:0 0 6px}} h2{{font-size:19px;margin:34px 0 12px;border-bottom:1px solid var(--line);padding-bottom:6px}}
.sub{{color:var(--mut);margin:0 0 16px}}
.panel{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin:14px 0}}
.charts{{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}} .chart{{background:var(--card2);border:1px solid var(--line);border-radius:10px;padding:8px 10px}} .ct{{font-size:12px;color:var(--mut);margin-bottom:4px}}
.tbl{{width:100%;border-collapse:collapse;font-size:13px}} .tbl th,.tbl td{{padding:6px 9px;border-bottom:1px solid var(--line);text-align:right}}
.tbl th:first-child,.tbl td:first-child{{text-align:left}} .tbl th{{color:var(--mut);font-weight:600}} .tbl .sep{{border-left:2px solid var(--line)}} td.pos{{color:#4fd18b}} td.neg{{color:#e6685f}}
.cards{{display:grid;grid-template-columns:repeat(auto-fill,minmax(560px,1fr));gap:14px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px 14px}} .card.edge{{border-color:#5a3a2a}}
.hdr{{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:6px}} .lbl{{font-weight:700;font-size:16px}} .kind{{font-size:10px;letter-spacing:.08em;color:var(--mut)}}
.srcemo{{font-size:11px;color:var(--mut);margin-bottom:8px}} .sub2{{font-size:11px;color:var(--mut);margin:10px 0 5px}}
.row,.row2{{display:grid;gap:8px}} .row2{{grid-template-columns:1fr 1fr}}
.cell{{background:var(--card2);border:1px solid var(--line);border-radius:9px;padding:8px 10px}} .cell.best{{border-color:var(--sid);box-shadow:0 0 0 1px var(--sid) inset}} .cell.rew{{border-color:var(--rew);box-shadow:0 0 0 1px var(--rew) inset}}
.ctag{{font-size:10px;color:var(--mut);letter-spacing:.03em;margin-bottom:5px;text-transform:uppercase}}
audio{{width:100%;height:32px}} .meta{{margin-top:6px;display:flex;flex-wrap:wrap;gap:4px;align-items:center}}
.badge{{font-size:10.5px;background:#232a34;border:1px solid var(--line);border-radius:6px;padding:1px 5px;color:var(--mut)}}
.badge b{{color:var(--fg)}} .badge.g b{{color:#4fd18b}} .badge.b b{{color:#6fa8ff}} .badge.p b{{color:#c39bff}} .badge.o b{{color:#f0a463}} .badge.r b{{color:#e9c74e}} .badge.r{{border-color:#5a5024}}
.note{{font-size:10px;color:var(--mut)}}
.chip{{display:inline-block;position:relative;font-size:10px;background:#232a34;border:1px solid var(--line);border-radius:5px;padding:1px 6px;margin:2px 3px 2px 0;color:var(--mut);overflow:hidden}} .chip .bar{{position:absolute;left:0;bottom:0;height:2px;background:var(--acc)}}
.k{{color:var(--acc);font-weight:700}} code{{background:#232a34;padding:1px 5px;border-radius:4px;font-size:12px}}
.big{{font-size:15px}} .hl{{color:var(--rew);font-weight:700}}
@media(max-width:640px){{.cards{{grid-template-columns:1fr}}.charts{{grid-template-columns:1fr}}.row2{{grid-template-columns:1fr}}}}
</style>
<div class=wrap>
<h1>Chatterbox VC — expressivity×quality reward, best-of-N &amp; SIDON</h1>
<p class=sub>Target voice <b>Measured Slavic Historian</b> (<code>emolia_c0542</code>). {len(ids)} sources
(40 intense·free EmoNet emotions + edge-case bursts) → <b>32 seed-varied</b> Chatterbox conversions,
each <b>SIDON-refined</b> and scored on emotion (Empathic-Insight), quality (DNSMOS-distilled) and
speaker similarity (ECAPA + Orange timbre). A <b>reward</b> then re-ranks candidates by how
<b>expressive AND clean</b> they are.</p>

<div class=panel><b class=hl>Reward.</b> For each candidate: <b>peak emotion intensity</b> = max over the 40
Empathic-Insight emotion experts; <b>Overall-Q</b> = DNSMOS-distilled quality. Both are
<b>z-normalized over the whole pool</b> ({32*len(ids)} candidates) and summed:
<code>reward = z(peak-emotion) + z(Overall-Q)</code>. The ⭐ pick in each card maximises it.
This favours takes that commit to the emotion without going noisy.</p></div>

<h2>1 · Reward ranking — listen</h2>
<p class=sub>Headline row per card = reward winner (Chatterbox vs. +SIDON ★), then best-by-Quality and best-by-emotion-MAE for reference.</p>
<h3 style='color:var(--mut);font-size:14px;margin:18px 0 8px'>40 intense · free-flowing emotions</h3>
<div class=cards>{emo_cards}</div>
<h3 style='color:var(--mut);font-size:14px;margin:26px 0 8px'>Edge cases — screams, whimper, crying, groan, shiver, laughter</h3>
<div class=cards>{edge_cards}</div>

<h2>2 · Diminishing returns — how many candidates?</h2>
<p class=sub>Expected best-of-k <b>reward</b> (+SIDON) across {len(ids)} groups, as % of the full 1→32 gain.
This reward deliberately chases the <b>most intense</b> take, so — unlike quality, which saturates by k≈8 —
it keeps climbing into the tail: <b>k=4 → {cap[4]:.0f}%</b>, <b>k=8 → {cap[8]:.0f}%</b>, <b>k=16 → {cap[16]:.0f}%</b>.
Recommended operating point <b class=hl>k={rec}</b> (marginal gain drops off after); the strict 90%-of-gain knee only lands at k={knee}.</p>
<div class=panel><div class=charts><div class=chart><div class=ct>Reward gain captured vs. k</div>{svg_dr()}</div>
<div class=chart><div class=ct>Marginal reward per step</div>
<table class=tbl><thead><tr><th>k</th><th>reward</th><th>captured</th><th>Δ vs prev</th></tr></thead><tbody>{dr_rows()}</tbody></table></div></div>
<p class=sub style='margin:10px 0 0'><b>k=4</b> already gets you halfway; <b>k=8</b> is the practical sweet spot
(~70% of the gain at ¼ the cost of 32). Cost is linear in k, so 16→32 pays 2× for the last ~13%.
If you only care about <b>quality</b>, k=8 fully saturates; if you're mining the <b>most extreme</b> expressive
takes, more candidates keep paying off slowly.</p></div>

<h2>3 · Cost — GPU-hours for 1,000,000 samples (SIDON + ranking)</h2>
<p class=sub>Measured per-candidate (ref clip {gph['clip_ref_s']:.1f}s output): generation <b>{pc['gen']*1000:.0f} ms</b>,
SIDON <b>{pc['sidon']*1000:.0f} ms</b>, scoring <b>{pc['score']*1000:.0f} ms</b> (all 40 emotion + quality experts, batched).
SIDON is the dominant cost, so the efficient design <b>ranks first, then SIDON only the winner</b> (×1 not ×N).</p>
<div class=panel><table class=tbl style='max-width:720px'><thead>
<tr><th rowspan=2>N candidates</th><th colspan=2 style='text-align:center'>efficient (rank → SIDON winner)</th><th colspan=2 style='text-align:center;border-left:2px solid var(--line)'>SIDON-all-then-rank</th></tr>
<tr><th>s / sample</th><th>GPU-hours / 1M</th><th class=sep>s / sample</th><th>GPU-hours / 1M</th></tr></thead>
<tbody>{gph_rows()}</tbody></table>
<p class=sub style='margin:12px 0 0'>At the recommended <b class=hl>N={rec}</b>: <b class=hl>{effK['gpu_hours']:,.0f} GPU-hours</b> per 1M with the efficient
pipeline (rank the {rec} candidates, then SIDON only the winner) vs <b>{allK['gpu_hours']:,.0f}</b> if you SIDON every
candidate — on 8×A100 that is ≈ <b>{effK['gpu_hours']/8/24:.1f} days</b> vs {allK['gpu_hours']/8/24:.1f} days wall-clock.
Since SIDON ({pc['sidon']*1000:.0f} ms) costs more than generation ({pc['gen']*1000:.0f} ms), refining only the winner
roughly <b>halves</b> the bill. Scoring all 40 emotion + quality experts is nearly free ({pc['score']*1000:.0f} ms,
batched). Estimate is compute-only (100% util); real runs add ~20–30% for I/O and mp3 encoding, and scale
~linearly with clip length ({gph['clip_ref_s']:.1f}s reference).</p></div>

<h2>4 · Global reward leaderboard (top 15, +SIDON pool)</h2>
<div class=panel><table class=tbl><thead><tr><th>#</th><th>source emotion</th><th>reward</th><th>peak-emo</th><th>strongest emotion</th><th>Ovr-Q</th></tr></thead><tbody>{top_rows()}</tbody></table></div>

<h2>5 · Best-of-k, SIDON effect &amp; throughput (reference)</h2>
<div class=panel><table class=tbl><thead>
<tr><th rowspan=2>k</th><th colspan=5 style='text-align:center'>RAW Chatterbox</th><th colspan=5 style='text-align:center;border-left:2px solid var(--line)'>+ SIDON</th></tr>
<tr><th>emo-cos</th><th>emo-MAE</th><th>Ovr-Q</th><th>ECAPA</th><th>Orange</th><th class=sep>emo-cos</th><th>emo-MAE</th><th>Ovr-Q</th><th>ECAPA</th><th>Orange</th></tr></thead>
<tbody>{bk_table()}</tbody></table></div>
<div class=panel><b>SIDON effect</b> (mean over {32*len(ids)} candidates) &nbsp; <span class=note>source→target identity baseline ECAPA {se:.2f} · Orange {so:.2f}</span>
<table class=tbl style='max-width:560px;margin-top:8px'><thead><tr><th>metric</th><th>RAW</th><th>+SIDON</th><th>Δ</th></tr></thead><tbody>{eff_rows()}</tbody></table></div>
<div class=panel><b>Throughput</b> — batched seeds, one A100. Mean {runstats['mean_gen_s_32']:.1f}s / group of 32, RTF {runstats['mean_rtf_b32']:.3f}.{rtf_html}</div>
</div>"""
open(OUT+"/index.html","w").write(HTML)
print("wrote index.html", f"({len(HTML)//1024} KB)", "sources", len(ids))
