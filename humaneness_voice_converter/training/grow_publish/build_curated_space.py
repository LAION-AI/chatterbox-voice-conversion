#!/usr/bin/env python3
"""Publish English-first curated VC page with original S3Gen comparison.

The browser reuses existing packed audio for the source and four tuned arms.
Only selected vanilla predictions are added, in eight packed JSON containers.
Selection depends on source quality proxies alone, never on model performance.
"""
from __future__ import annotations

import base64
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import tarfile

from huggingface_hub import CommitOperationAdd, HfApi
import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
OUT = Path('/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_followup_eval')
VANILLA = OUT / 'vanilla_curation'
STAGE = OUT / 'space_stage'
REPO = 'laion/chatterbox-s3gen-vc-grow-24k-listening'
ARMS = ('vanilla', 'baseline', 'continue', 'remerged_new', 'speaker_mix')
LABELS = {'vanilla': 'Vanilla Chatterbox · CAMPPlus only',
          'baseline': '300h supervised baseline',
          'continue': 'RL Continue · CE/emotion',
          'remerged_new': 'RL Re-merged · CE/emotion',
          'speaker_mix': 'RL Speaker-Mix · CE/emotion/identity'}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def load_records() -> dict[str, dict]:
    proofs = [read_json(VANILLA / f'vanilla-rank{r}.COMPLETE.json') for r in range(4)]
    assert all(p['status'] == 'COMPLETE' and p['sources'] == 40 and
               p['predictions'] == 80 and p['vanilla_no_lora'] and
               p['vanilla_no_style_projection'] for p in proofs)
    records = {}
    for rank in range(4):
        for line in (VANILLA / f'vanilla-rank{rank}.jsonl').read_text().splitlines():
            record = json.loads(line)
            assert record['case_id'] not in records
            records[record['case_id']] = record
    assert len(records) == 160
    return records


def choose(cases: list[dict], quality: dict[str, dict]) -> tuple[list[dict], dict]:
    by_emotion = defaultdict(list)
    for case in cases:
        by_emotion[int(case['emotion_index'])].append(case)
    assert len(by_emotion) == 40
    selected = []
    threshold = {'predicted_p808_mos': 3.20, 'camp_window_min_cosine': 0.40}

    def acceptable(case: dict) -> bool:
        row = quality[case['case_id']]
        return (row['source_predicted_p808_mos'] >= threshold['predicted_p808_mos']
                and row['source_camp_window_min_cosine'] >= threshold['camp_window_min_cosine'])

    def rank(case: dict) -> tuple:
        row = quality[case['case_id']]
        # Acoustic quality dominates; known RL-source overlap is a modest
        # penalty, not grounds to choose a clearly worse recording.
        return (row['source_predicted_p808_mos'] +
                .35 * row['source_camp_window_min_cosine'] +
                .025 * row['source_audiobox_pq'] -
                .12 * bool(case['rl_training_uid_overlap']),
                not case['rl_training_uid_overlap'],
                case['origin'] == 'real')

    for index in range(40):
        english = [case for case in by_emotion[index] if case['lang'] == 'en']
        german = [case for case in by_emotion[index] if case['lang'] == 'de']
        assert len(english) == len(german) == 2
        passing_en = [case for case in english if acceptable(case)]
        passing_de = [case for case in german if acceptable(case)]
        # English gets priority if it passes the acoustic gate. German is a
        # fallback for emotions where neither available English input does.
        candidates = passing_en or passing_de or (english + german)
        chosen = max(candidates, key=rank)
        selected.append(chosen | {'quality_gate_pass': acceptable(chosen),
                                  'curation_group': ('English-first' if chosen['lang'] == 'en'
                                                     else 'German quality fallback')})
    assert len(selected) == 40 and len({case['case_id'] for case in selected}) == 40
    selected.sort(key=lambda c: (c['lang'] != 'en', int(c['emotion_index']), int(c['sample'])))
    audit = {'strategy': 'one input-quality-selected source per emotion; English preferred, German used if both English candidates fail',
             'thresholds': threshold,
             'english_cases': sum(case['lang'] == 'en' for case in selected),
             'german_cases': sum(case['lang'] == 'de' for case in selected),
             'quality_gate_failures': sum(not case['quality_gate_pass'] for case in selected),
             'rl_source_overlap_cases': sum(case['rl_training_uid_overlap'] for case in selected),
             'selected_case_ids': [case['case_id'] for case in selected],
             'selection_features': ['predicted P808 MOS', 'CAMPPlus 2-second-window minimum cosine',
                                    'Audiobox PQ', 'known RL source UID overlap'],
             'warning': 'Predicted MOS is not a human mean opinion score; window cosine is not diarization. Multi-speaker clips may remain.'}
    return selected, audit


def build() -> dict:
    assert read_json(OUT / 'SPACE_PUBLISHED.json')['status'] == 'COMPLETE'
    site = read_json(OUT / 'site_inputs.json')
    prep = {row['case_id']: row for row in pq.read_table(
        OUT / 'prepared.parquet', columns=['case_id', 'role', 'rl_training_uid_overlap']).to_pylist()
            if row['role'] == 'source'}
    cases = [case | {'rl_training_uid_overlap': bool(prep[case['case_id']]['rl_training_uid_overlap'])}
             for case in site['cases']]
    vanilla = load_records()
    selected, audit = choose(cases, vanilla)
    metrics_rows = pq.read_table(OUT / 'metrics.parquet').to_pylist()
    metric_map = {(row['case_id'], row['arm'], row['target_label']): row
                  for row in metrics_rows}
    metric_out = {}
    locations = {}
    selected_public = []
    selected_set = {case['case_id'] for case in selected}
    for case in selected:
        cid, index = case['case_id'], int(case['emotion_index'])
        audio_source = f'{cid}/original.mp3'
        locations[audio_source] = f'packs/emotion-{index:02d}.json'
        selected_public.append({key: case[key] for key in
                               ('case_id', 'uid', 'emotion', 'emotion_index', 'lang',
                                'origin', 'text', 'sample', 'rl_training_uid_overlap',
                                'quality_gate_pass', 'curation_group')}
                               | {'source_quality': {key: vanilla[cid][key] for key in
                                     ('source_audiobox_ce', 'source_audiobox_pq',
                                      'source_predicted_p808_mos', 'source_camp_window_min_cosine')}})
        for label in ('A', 'B'):
            for arm in ARMS:
                member = f'{cid}/{arm}_vc_{label}.mp3'
                if arm == 'vanilla':
                    row = vanilla[cid]['predictions'][label]
                    assert row['mp3_member'] == member
                    locations[member] = f'packs/curated-vanilla-{index // 5:02d}.json'
                else:
                    row = metric_map[(cid, arm, label)]
                    locations[member] = f'packs/emotion-{index:02d}.json'
                metric_out[member] = {key: row[key] for key in
                                      ('audiobox_ce', 'audiobox_pq',
                                       'top2_emotion_mse', 'target_camp_cosine')}
    assert len(locations) == len(selected) * (1 + 2 * len(ARMS))

    vanilla_tars = [tarfile.open(VANILLA / f'vanilla-rank{rank}.tar', 'r') for rank in range(4)]
    vanilla_pack_files = []
    try:
        for group in range(8):
            pack = {}
            for case in selected:
                if int(case['emotion_index']) // 5 != group:
                    continue
                cid = case['case_id']
                for label in ('A', 'B'):
                    member = vanilla[cid]['predictions'][label]['mp3_member']
                    payload = None
                    for archive in vanilla_tars:
                        try:
                            payload = archive.extractfile(member).read()
                            break
                        except KeyError:
                            pass
                    assert payload is not None, member
                    pack[member] = base64.b64encode(payload).decode('ascii')
            path = STAGE / f'packs/curated-vanilla-{group:02d}.json'
            atomic(path, (json.dumps(pack, separators=(',', ':')) + '\n').encode())
            vanilla_pack_files.append(path)
    finally:
        for archive in vanilla_tars:
            archive.close()
    with tarfile.open(OUT / 'site_inputs.tar', 'r') as archive:
        references = {f'references/{label}.mp3': base64.b64encode(
            archive.extractfile(f'references/{label}.mp3').read()).decode('ascii')
                      for label in ('A', 'B')}
    summary = {}
    for arm in ARMS:
        arm_rows = [metric_out[f'{case["case_id"]}/{arm}_vc_{label}.mp3']
                    for case in selected for label in ('A', 'B')]
        summary[arm] = {key: float(np.mean([row[key] for row in arm_rows]))
                        for key in ('audiobox_ce', 'audiobox_pq',
                                    'top2_emotion_mse', 'target_camp_cosine')}
    data = {'schema': 'chatterbox-s3gen-vc-curated-vanilla-v1', 'cases': selected_public,
            'targets': [{key: target[key] for key in ('label', 'lang', 'text', 'uid')}
                        for target in site['targets']],
            'references': references, 'locations': locations, 'metrics': metric_out,
            'arms': ARMS, 'labels': LABELS, 'summary': summary,
            'audit': audit, 'selected_steps': read_json(OUT / 'EVAL_DONE.json')['selected_steps']}
    template = (HERE / 'curated_template.html').read_text()
    assert template.count('__DATA_JSON__') == 1
    compact = json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c')
    html = template.replace('__DATA_JSON__', compact)
    page = STAGE / 'curated.html'
    atomic(page, html.encode('utf-8'))
    index_path = STAGE / 'index.html'
    index_html = index_path.read_text()
    if 'href="curated.html"' not in index_html:
        anchor = '<h1>One source. Two voices. Four models.</h1>'
        assert index_html.count(anchor) == 1
        index_html = index_html.replace(
            anchor, anchor + '<p><a href="curated.html">Open the English-first curated comparison, including untouched vanilla Chatterbox →</a></p>')
        atomic(index_path, index_html.encode('utf-8'))
    readme_path = STAGE / 'README.md'
    readme = readme_path.read_text()
    if '[English-first curated comparison](curated.html)' not in readme:
        readme += ('\n## English-first curated comparison\n\n'
                   'The [English-first curated comparison](curated.html) adds the untouched '
                   'pretrained Chatterbox S3Gen (CAMPPlus-only, no VoiceNet/EmoNet '
                   'projection) beside the four previously evaluated arms. One source '
                   'per emotion was chosen using input-audio quality proxies, with '
                   'English preferred and German used as a quality fallback. Selection and limitations '
                   'are documented in [CURATED.json](CURATED.json) and on the page.\n')
        atomic(readme_path, readme.encode('utf-8'))
    atomic(STAGE / 'CURATED.json', (json.dumps(audit | {'summary': summary}, indent=2) + '\n').encode())
    build_proof = {'status': 'COMPLETE', 'page': 'curated.html', 'cases': len(selected),
                   'vanilla_predictions_published': len(selected) * 2,
                   'new_packed_files': len(vanilla_pack_files),
                   'page_sha256': hashlib.sha256(page.read_bytes()).hexdigest(),
                   'quality_gate_failures': audit['quality_gate_failures']}
    atomic(VANILLA / 'BUILD_DONE.json', (json.dumps(build_proof, indent=2) + '\n').encode())
    return build_proof


def publish() -> None:
    api = HfApi()
    paths = [STAGE / f'packs/curated-vanilla-{i:02d}.json' for i in range(8)]
    paths += [STAGE / 'CURATED.json', STAGE / 'curated.html',
              STAGE / 'README.md', STAGE / 'index.html']
    assert all(path.exists() for path in paths)
    ops = [CommitOperationAdd(path_in_repo=str(path.relative_to(STAGE)),
                              path_or_fileobj=str(path)) for path in paths]
    api.create_commit(repo_id=REPO, repo_type='space', operations=ops,
                      commit_message='Add English-first curated vanilla VC comparison')
    remote = api.repo_info(repo_id=REPO, repo_type='space', files_metadata=True)
    files = {file.rfilename: file for file in remote.siblings}
    for path in paths:
        name = str(path.relative_to(STAGE))
        assert name in files and files[name].size == path.stat().st_size, name
        if files[name].lfs:
            assert files[name].lfs.sha256 == hashlib.sha256(path.read_bytes()).hexdigest(), name
    proof = read_json(VANILLA / 'BUILD_DONE.json')
    proof['url'] = 'https://laion-chatterbox-s3gen-vc-grow-24k-listening.static.hf.space/curated.html'
    proof['published_revision'] = remote.sha
    atomic(VANILLA / 'PUBLISHED.json', (json.dumps(proof, indent=2) + '\n').encode())
    print(json.dumps(proof), flush=True)


if __name__ == '__main__':
    proof = build()
    print(json.dumps(proof), flush=True)
    publish()
