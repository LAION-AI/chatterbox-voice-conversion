#!/usr/bin/env python3
"""Build the standalone Humaneness Voice Converter static demo and audio manifest.

The build reads existing packed evaluation artifacts; it never synthesizes
audio and never creates per-sample local files. The upload helper streams
small MP3 bytes directly from the existing packs/TARs to a HF dataset repo.
"""
from __future__ import annotations

import base64
from collections import defaultdict
import html
import io
import json
from pathlib import Path
import re
import tarfile

import pyarrow.parquet as pq

ROOT = Path('/e/scratch/laionize/schuhmann1')
ORIGINAL = ROOT / 'chatterbox_s3gen_grow_followup_eval/space_stage'
BON = ROOT / 'hvc_best_of_n_2026_10_11/full'
EXTERNAL = ROOT / 'hvc_benchmark_2026_10_11/evaluation'
STAGE = ROOT / 'humaneness_voice_converter_demo_stage'
AUDIO_REPO = 'laion/Humaneness-Voice-Converter-Demo-Audio'
MODEL_REPO = 'laion/Humaneness-Voice-Converter'
SPACE_REPO = 'laion/Humaneness-Voice-Converter-Demo'
ARMS = ('vanilla', 'full40k', 'seed_v1', 'seed_v2', 'xvc', 'bon4', 'bon8', 'bon16')
LABELS = {
    'vanilla': 'Vanilla Chatterbox', 'full40k': 'Humaneness 40k',
    'seed_v1': 'Seed-VC v1', 'seed_v2': 'Seed-VC v2', 'xvc': 'X-VC (2026)',
    'bon4': 'Humaneness Best-of-4', 'bon8': 'Humaneness Best-of-8',
    'bon16': 'Humaneness Best-of-16',
}
FIELDS = ('audiobox_ce', 'audiobox_pq', 'top2_emotion_mse', 'target_camp_cosine')


def old_data():
    page = (ORIGINAL / 'curated.html').read_text()
    match = re.search(r'<script id="eval-data" type="application/json">(.*?)</script>',
                      page, re.S)
    assert match
    return json.loads(match.group(1))


def read_old_audio(data, member, cache):
    filename = data['locations'][member]
    if filename not in cache:
        cache[filename] = json.loads((ORIGINAL / filename).read_text())
    return base64.b64decode(cache[filename][member])


def read_tars(directory, prefix):
    found = {}
    for rank in range(4):
        with tarfile.open(directory / f'{prefix}-rank{rank}.tar', 'r') as archive:
            for item in archive:
                assert item.isfile() and item.name not in found and item.name.endswith('.mp3')
                with archive.extractfile(item) as handle:
                    found[item.name] = handle.read()
    return found


def mean_records(rows):
    return {key: sum(float(row[key]) for row in rows) / len(rows) for key in FIELDS}


def make():
    data = old_data()
    selected = {case['case_id'] for case in data['cases']}
    assert len(selected) == 40
    licenses = {row['case_id']: row for row in json.loads((ORIGINAL / 'INPUTS.json').read_text())['cases']}
    assert {licenses[cid]['license'] for cid in selected} == {'CC-BY-4.0'}
    old_cache = {}
    archives = {'bon': read_tars(BON, 'winners')}
    for arm in ('seed_v1', 'seed_v2', 'xvc'):
        archives[arm] = read_tars(EXTERNAL / arm / 'full', 'audio')
    bon_rows = pq.read_table(BON / 'all_winners.parquet').to_pylist()
    assert len(bon_rows) == 320
    ext_rows = {arm: pq.read_table(EXTERNAL / arm / 'full/metrics.parquet').to_pylist()
                for arm in ('seed_v1', 'seed_v2', 'xvc')}
    assert all(len(rows) == 80 for rows in ext_rows.values())
    summaries = {arm: data['summary'][arm] for arm in ('vanilla', 'full40k')}
    summaries.update({arm: mean_records(rows) for arm, rows in ext_rows.items()})
    for n in (4, 8, 16):
        summaries[f'bon{n}'] = mean_records([row for row in bon_rows if row['pool_size'] == n])
    metrics = {arm: {} for arm in ARMS}
    for row in bon_rows:
        if row['pool_size'] in (4, 8, 16):
            metrics[f"bon{row['pool_size']}"][(row['case_id'], row['target_label'])] = row
    for arm, rows in ext_rows.items():
        metrics[arm] = {(row['case_id'], row['target_label']): row for row in rows}
    for arm in ('vanilla', 'full40k'):
        for cid in selected:
            for label in ('A', 'B'):
                metrics[arm][(cid, label)] = data['metrics'][f'{cid}/{arm}_vc_{label}.mp3']
    mp3 = {}
    cases = []
    for case in data['cases']:
        cid = case['case_id']
        record = {key: case[key] for key in ('case_id', 'uid', 'emotion', 'lang',
                                             'origin', 'text', 'source_quality',
                                             'rl_training_uid_overlap')}
        record['license'] = licenses[cid]['license']
        record['hf_source'] = licenses[cid]['hf_source']
        record['audio'] = {'original': f'audio/{cid}/original.mp3'}
        mp3[record['audio']['original']] = read_old_audio(data, f'{cid}/original.mp3', old_cache)
        record['arms'] = {}
        for arm in ARMS:
            record['arms'][arm] = {}
            for label in ('A', 'B'):
                member = f'{cid}/{arm}_vc_{label}.mp3'
                path = f'audio/{member}'
                if arm in ('vanilla', 'full40k'):
                    audio = read_old_audio(data, member, old_cache)
                elif arm.startswith('bon'):
                    audio = archives['bon'][member]
                else:
                    audio = archives[arm][member]
                mp3[path] = audio
                row = metrics[arm][(cid, label)]
                record['arms'][arm][label] = {'path': path,
                    'metrics': {field: float(row[field]) for field in FIELDS}}
        cases.append(record)
    refs = {}
    for target in data['targets']:
        label = target['label']
        path = f'audio/references/{label}.mp3'
        mp3[path] = base64.b64decode(data['references'][f'references/{label}.mp3'])
        refs[label] = {'path': path, 'text': target['text'], 'lang': target['lang'],
                       'uid': target['uid']}
    assert len(mp3) == 40 * (1 + 2 * len(ARMS)) + 2 == 682
    assert all(len(audio) > 1000 and audio[:3] == b'ID3' for audio in mp3.values())
    bon_done = json.loads((BON / 'DONE.json').read_text())
    manifest = {'schema': 'humaneness-vc-standalone-demo-v1',
                'model_repo': MODEL_REPO, 'audio_repo': AUDIO_REPO,
                'cases': cases, 'references': refs, 'arms': ARMS,
                'labels': LABELS, 'summary': summaries,
                'best_of_n': {str(n): bon_done['summary'][str(n)] for n in (1, 4, 8, 16)},
                'audio_count': len(mp3),
                'method': {'cases': 40, 'targets': 2, 'english_cases': 32,
                           'german_cases': 8, 'rl_training_uid_overlap_cases': 10,
                           'model': '40k full-flow after 24k Speaker-Mix',
                           'proxy_judge': 'Humaneness Ears Medium for CE/PQ/emotions; CAMPPlus for target cosine',
                           'external_models': 'official Seed-VC v1/v2 and X-VC checkpoints; same sources and targets',
                           'caveat': 'Not a blinded subjective, independent judge, WER or speaker-disjoint study.'}}
    STAGE.mkdir(parents=True, exist_ok=True)
    (STAGE / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    (STAGE / 'index.html').write_text(page_html(manifest))
    (STAGE / 'README.md').write_text(space_readme())
    audio_readme = '''---
license: cc-by-4.0
pretty_name: Humaneness Voice Converter Demo Audio
tags:
  - audio
  - voice-conversion
---

# Humaneness Voice Converter Demo Audio

682 MP3s for the [standalone listening demo](https://huggingface.co/spaces/laion/Humaneness-Voice-Converter-Demo): 40 original sources, two fixed target references, and 40 × 2 outputs for each of Vanilla Chatterbox, Humaneness 40k, Seed-VC v1, Seed-VC v2, X-VC, and Humaneness Best-of-4/8/16. All are 24-kHz mono 128-kb/s MP3. A CSV-like schema and per-case metrics are in the Space `manifest.json`. No model weights or external source code are included here.

The selected source clips were marked CC BY 4.0 in the source inventory; target A is CC0 and target B is CC BY 4.0. Generated outputs retain the rights and attribution obligations of their source recordings. Provenance is in the Space manifest. External models retain their own licenses and are credited in the Space. Use only with authorization from the speakers and rightsholders.
'''
    (STAGE / 'AUDIO_README.md').write_text(audio_readme)
    index = {path: {'bytes': len(content)} for path, content in mp3.items()}
    (STAGE / 'AUDIO_INDEX.json').write_text(json.dumps(index, indent=2) + '\n')
    print(json.dumps({'status': 'BUILT', 'stage': str(STAGE),
                      'mp3_count': len(mp3), 'mp3_bytes': sum(map(len, mp3.values())),
                      'html_bytes': (STAGE / 'index.html').stat().st_size}))
    return mp3


def page_html(d):
    summary_rows = ''.join(
        '<tr><td>'+html.escape(d['labels'][arm])+'</td>' + ''.join(
            f'<td>{d["summary"][arm][field]:.3f}</td>' for field in FIELDS) + '</tr>'
        for arm in d['arms'])
    encoded = json.dumps(d, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Humaneness Voice Converter Demo</title>
<style>:root{{--bg:#f4f5f3;--fg:#17262d;--mut:#62737b;--line:#dce4e1;--accent:#b34d31}}*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}}main{{max-width:1650px;margin:auto;padding:22px}}h1{{font-size:clamp(30px,4vw,52px);line-height:1.08;margin:5px 0}}h2{{font-size:20px;margin:0 0 8px}}p{{margin:7px 0}}a{{color:var(--accent)}}.lead{{font-size:17px;max-width:1050px}}.mut{{color:var(--mut)}}.card{{background:white;border:1px solid var(--line);border-radius:13px;padding:16px 18px;margin:14px 0}}table{{border-collapse:collapse;width:100%;font-size:13px}}td,th{{padding:8px;border-bottom:1px solid var(--line);text-align:right}}td:first-child,th:first-child{{text-align:left}}.scroll{{overflow:auto}}.tools{{position:sticky;top:0;z-index:2;background:#f4f5f3ef;display:flex;gap:10px;flex-wrap:wrap;padding:12px 0}}select,input{{padding:8px 10px;border:1px solid var(--line);border-radius:8px;background:white;font:inherit}}.case{{background:white;border:1px solid var(--line);border-radius:13px;margin:13px 0;padding:15px}}.badge{{display:inline-block;padding:1px 6px;background:#edf1f0;border-radius:5px;font-size:11px;margin:3px}}.players{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin-top:10px}}.player{{border:1px solid var(--line);border-radius:8px;padding:9px}}.player b{{display:block;font-size:12px}}.player audio{{width:100%}}.measure{{font-size:11px;color:var(--mut)}}.refgrid{{display:flex;gap:18px;flex-wrap:wrap}}.refgrid audio{{width:250px}}.group{{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--mut)}}details summary{{cursor:pointer;font-weight:700}}.hidden{{display:none}}</style></head><body><main><p class="group">LAION · speech-to-speech research demo</p><h1>Humaneness Voice Converter</h1>
<p class="lead">One source recording, one target voice. Compare the original source, untouched Chatterbox, the released 40k Humaneness flow decoder, three independently run voice converters, and optional Best-of-4/8/16 selections. All columns use the same 40 curated sources and two fixed targets.</p>
<p><a href="https://huggingface.co/{MODEL_REPO}">Model and tested inference code</a> · <a href="https://huggingface.co/datasets/{AUDIO_REPO}">Individual MP3 files</a> · <a href="manifest.json">Manifest and per-case scores</a></p>
<section class="card"><h2>What these numbers mean</h2><p class="mut">40 source clips (32 English, 8 German) × two targets. Higher AudioBox Content Enjoyment (CE), Production Quality (PQ) and target-speaker CAMPPlus cosine are directionally better; lower source-top-two-emotion MSE is better. CE/PQ/emotion are Humaneness Ears Medium model estimates, <b>not human MOS or an independent emotion judge</b>. Ten curated source UIDs overlap known RL source UIDs; speaker-disjointness was not established. One source did not pass the soft input-quality gate. Listen rather than treating a proxy score as proof.</p><div class="scroll"><table><thead><tr><th>Converter / selection</th><th>CE ↑</th><th>PQ ↑</th><th>Emotion MSE ↓</th><th>Target cosine ↑</th></tr></thead><tbody>{summary_rows}</tbody></table></div><p class="mut">The 40k model improves CE and PQ over Vanilla on this set, but <b>Vanilla has higher target-speaker cosine</b>. Seed-VC v1 has the highest target cosine of the currently tested external arms. Best-of-N is chosen by an equal-third standardized score of CE, speaker cosine and negative emotion MSE, calibrated on 93 separate noncurated source cases. Best-of-N and the 40k model use the same Ears family as training, so selection can favor that judge.</p></section>
<section class="card"><h2>Best-of-N: quality versus latency</h2><p>For each source/target pair we generated 16 distinct stochastic takes and selected the highest-reward take from nested pools of 1, 4, 8 and 16. The original <b>sequential</b> evaluation spent about 0.73 s, 2.69 s, 5.31 s and 10.54 s per pair for N=1/4/8/16, including candidate scoring. The selected reward rises from 0.322 to 0.366, 0.387 and 0.399; speaker cosine rises from 0.755 to 0.766, 0.772 and 0.776. The reward necessarily rises with nested pools, but individual CE/PQ need not.</p><p>We also tested <b>true GPU batching</b> on this 40k checkpoint and one GH200. For a 6.53-s source, median warmed decoder+HiFT wall times in seconds were:</p><div class="scroll"><table><thead><tr><th>Precision / candidate pool</th><th>N=1</th><th>N=4</th><th>N=8</th><th>N=16</th></tr></thead><tbody><tr><td>FP32 batch</td><td>0.468</td><td>0.526</td><td>0.831</td><td>1.534</td></tr><tr><td>BF16 batch</td><td>0.558</td><td>0.605</td><td>0.643</td><td>0.665</td></tr><tr><td>Experimental FP8 estimator Linear layers only</td><td>4.373</td><td>5.074</td><td>5.097</td><td>5.038</td></tr></tbody></table></div><p class="mut">Three warmed repeats, fixed 10 flow steps, cached source tokens/reference/scores; model load, tokenization, Ears/CAMPPlus ranking, MP3 and network excluded. FP32 was faster for this single/4-take short case; BF16 won clearly for 8/16 takes. FP8 quantized 352 Linear modules via TorchAO 0.15 but required removing incompatible upstream inference-mode wrappers and was <b>much slower</b>; we do not recommend it. These N candidates are parallel candidate streams for <b>one eventual winner</b>, not N simultaneous users. A winner still waits for selection and audio preparation. In completed end-to-end CLI smokes on this clip, the 4-take path spent 0.616 s generating, 0.135 s ranking and 6.351 s preparing; one 16-take batch spent 0.752 s generating, 0.261 s ranking and 6.283 s preparing. See the <a href="https://huggingface.co/{MODEL_REPO}/blob/main/SPEED.md">full method and all three clip lengths</a>.</p><p class="mut">The <a href="https://github.com/LAION-AI/chatterbox-voice-conversion#speed-optimization-bf16-flow-and-true-gpu-batching">upstream Chatterbox VC recipe</a> reports separate RTX 3090 measurements for Vanilla S3Gen; those cannot be substituted for our GH200/40k measurements.</p></section>
<section class="card"><h2>Target voices</h2><div class="refgrid" id="references"></div><p class="mut">Target A is German (CC0); target B is English (CC BY 4.0). A target voice is an acoustic example, not proof of identity or consent.</p></section>
<details class="card"><summary>Evaluation method, source rights and limitations</summary><p>Official Seed-VC v1/v2 and X-VC checkpoints were run on the same source files and target references, with their official/offline inference paths. The comparisons use one generation per pair for each external arm. The 40k and Best-of-N arms use the completed full-flow checkpoint. Each displayed audio file is 24-kHz mono, 128-kb/s MP3 in the linked dataset. For per-case provenance, seed, proxy metric and model details see the manifest and model card. The original source selections are marked CC BY 4.0. External models keep their respective licenses; their weights/code are not hosted here. This is not a WER benchmark, a blinded preference trial, a human MOS study or verified speaker identity test.</p></details>
<div class="tools"><select id="target"><option value="B">Target B · English</option><option value="A">Target A · German</option></select><select id="language"><option value="all">English first + German</option><option value="en">English only</option><option value="de">German only</option></select><select id="emotion"><option value="all">All emotions</option></select><input id="search" placeholder="Search transcript or emotion" type="search"><label><input id="bestof" type="checkbox"> Show Best-of-N</label></div><p id="count" class="mut"></p><div id="grid"></div><p class="mut">Research preview. Obtain consent and authorization for target voices; do not use generated speech for impersonation or deception.</p></main>
<script id="demo-data" type="application/json">{encoded}</script><script>
const D=JSON.parse(document.getElementById('demo-data').textContent),base='https://huggingface.co/datasets/'+D.audio_repo+'/resolve/main/';let target='B',language='all',emotion='all',query='',showBest=false;
const el=(t,text,cls)=>{{let x=document.createElement(t);if(text!==undefined)x.textContent=text;if(cls)x.className=cls;return x}};
function player(path,label,m){{let p=el('div',undefined,'player'),title=el('b',label),a=el('audio');a.controls=true;a.preload='none';a.dataset.src=base+path;p.append(title,a);if(m)p.append(el('div','CE '+m.audiobox_ce.toFixed(2)+' · PQ '+m.audiobox_pq.toFixed(2)+' · MSE '+m.top2_emotion_mse.toFixed(3)+' · voice '+m.target_camp_cosine.toFixed(3),'measure'));return p}}
const observer=new IntersectionObserver(entries=>{{for(const e of entries){{if(!e.isIntersecting)continue;observer.unobserve(e.target);for(const a of e.target.querySelectorAll('audio[data-src]'))a.src=a.dataset.src}}}},{{rootMargin:'400px'}});
function draw(){{const g=document.getElementById('grid');g.replaceChildren();let count=0;for(const c of D.cases){{if(language!=='all'&&c.lang!==language)continue;if(emotion!=='all'&&c.emotion!==emotion)continue;if(query&&!(c.emotion+' '+c.text).toLowerCase().includes(query))continue;count++;let box=el('section',undefined,'case');box.append(el('h2',c.emotion),el('p',c.text));let meta=el('div');for(const s of [c.lang.toUpperCase(),c.origin,c.case_id,c.license])meta.append(el('span',s,'badge'));if(c.rl_training_uid_overlap)meta.append(el('span','RL-source UID overlap','badge'));box.append(meta);let ps=el('div',undefined,'players');ps.append(player(c.audio.original,'Original source'));for(const arm of D.arms){{if(arm.startsWith('bon')&&!showBest)continue;let item=c.arms[arm][target];ps.append(player(item.path,D.labels[arm],item.metrics))}}box.append(ps);g.append(box);observer.observe(box)}}document.getElementById('count').textContent=count+' of 40 cases · target '+target}}
for(const [label,r] of Object.entries(D.references)){{let b=el('div'),a=el('audio');a.controls=true;a.preload='none';a.src=base+r.path;b.append(el('b','Target '+label+' · '+r.lang.toUpperCase()),el('p',r.text),a);document.getElementById('references').append(b)}}
for(const name of [...new Set(D.cases.map(x=>x.emotion))]){{let o=el('option',name);o.value=name;document.getElementById('emotion').append(o)}}
for(const id of ['target','language','emotion'])document.getElementById(id).addEventListener('change',e=>{{if(id==='target')target=e.target.value;else if(id==='language')language=e.target.value;else emotion=e.target.value;draw()}});document.getElementById('search').addEventListener('input',e=>{{query=e.target.value.toLowerCase();draw()}});document.getElementById('bestof').addEventListener('change',e=>{{showBest=e.target.checked;draw()}});draw();
</script></body></html>'''


def space_readme():
    return '''---
title: Humaneness Voice Converter Demo
emoji: 🎙️
colorFrom: red
colorTo: blue
sdk: static
app_file: index.html
pinned: false
license: cc-by-4.0
---

Standalone static listening demo for [Humaneness Voice Converter](https://huggingface.co/laion/Humaneness-Voice-Converter). The audio files are [individual MP3s](https://huggingface.co/datasets/laion/Humaneness-Voice-Converter-Demo-Audio), loaded lazily. This Space contains no model weights, audio generation service or intermediate checkpoints. Input rights and metric caveats are explained on the page.
'''


if __name__ == '__main__':
    make()
