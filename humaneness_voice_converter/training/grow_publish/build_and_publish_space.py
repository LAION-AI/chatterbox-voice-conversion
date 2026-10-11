#!/usr/bin/env python3
"""Build and publish the packed 160-case HF Static Space after EVAL_DONE.

The audio is stored in 40 emotion-level JSON containers and lazy-loaded by
the browser. Upload is idempotent, with per-file size and available LFS SHA
verification. The index page is uploaded last, after every audio pack.
"""
from __future__ import annotations

import base64
from collections import defaultdict
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile

from huggingface_hub import HfApi
import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
OUT = Path('/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_followup_eval')
STAGE = OUT / 'space_stage'
REPO = 'laion/chatterbox-s3gen-vc-grow-24k-listening'
ARMS = ('baseline', 'continue', 'remerged_new', 'speaker_mix')
LABELS = {'baseline': '300h supervised baseline', 'continue': 'Continue · CE/emotion',
          'remerged_new': 'Re-merged · CE/emotion',
          'speaker_mix': 'Speaker-Mix · CE/emotion/identity'}
STATE = OUT / 'space_upload_state.json'


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_bytes(data)
    os.replace(temporary, path)


def fmean(rows: list[dict], key: str) -> float:
    return float(np.mean([row[key] for row in rows]))


def build() -> None:
    done = read_json(OUT / 'EVAL_DONE.json')
    assert done['status'] == 'COMPLETE' and done['cases'] == 160
    selection = read_json(OUT / 'SELECTED.json')
    selected = set(selection['selection_case_ids'])
    assert len(selected) == 40
    site = read_json(OUT / 'site_inputs.json')
    cases = site['cases']
    assert len(cases) == 160
    prepared = {row['case_id']: row for row in pq.read_table(
        OUT / 'prepared.parquet', columns=['case_id', 'rl_training_uid_overlap', 'role'])
                .to_pylist() if row['role'] == 'source'}
    metrics = pq.read_table(OUT / 'metrics.parquet').to_pylist()
    assert len(metrics) == 160 * 2 * len(ARMS)
    measure_lookup = {(row['case_id'], row['arm'], row['target_label']): row
                      for row in metrics}
    assert len(measure_lookup) == len(metrics)
    test_rows = [row for row in metrics if not row['rl_training_uid_overlap']
                 and row['case_id'] not in selected]
    assert len({row['case_id'] for row in test_rows}) == 83, len(test_rows)
    test_metrics = {arm: {key: fmean([row for row in test_rows if row['arm'] == arm], key)
                          for key in ('audiobox_ce', 'audiobox_pq',
                                      'top2_emotion_mse', 'target_camp_cosine')}
                    for arm in ARMS}
    by_emotion = defaultdict(list)
    for case in cases:
        by_emotion[int(case['emotion_index'])].append(case)
    assert len(by_emotion) == 40 and all(len(group) == 4 for group in by_emotion.values())
    groups = []
    locations = {}
    metric_js = {}
    with tarfile.open(OUT / 'site_inputs.tar', 'r') as originals:
        archives = [tarfile.open(OUT / f'audio-rank{rank}.tar', 'r') for rank in range(4)]
        try:
            references = {f'references/{label}.mp3': base64.b64encode(
                originals.extractfile(f'references/{label}.mp3').read()).decode('ascii')
                          for label in ('A', 'B')}
            for index in range(40):
                pack = {}
                public_cases = []
                for case in sorted(by_emotion[index], key=lambda x: x['sample']):
                    cid = case['case_id']
                    source_member = f'sources/{cid}.mp3'
                    original_member = f'{cid}/original.mp3'
                    pack[original_member] = base64.b64encode(
                        originals.extractfile(source_member).read()).decode('ascii')
                    for arm in ARMS:
                        for label in ('A', 'B'):
                            row = measure_lookup[(cid, arm, label)]
                            member = row['mp3_member']
                            audio = None
                            for archive in archives:
                                try:
                                    audio = archive.extractfile(member).read()
                                    break
                                except KeyError:
                                    continue
                            assert audio is not None, member
                            pack[member] = base64.b64encode(audio).decode('ascii')
                            metric_js[member] = {key: row[key] for key in
                                                 ('audiobox_ce', 'audiobox_pq',
                                                  'top2_emotion_mse', 'target_camp_cosine')}
                    public_cases.append({key: case[key] for key in
                                         ('uid', 'case_id', 'emotion', 'lang', 'origin',
                                          'text', 'emotion_index', 'sample')}
                                        | {'rl_training_uid_overlap': bool(
                                            prepared[cid]['rl_training_uid_overlap'])})
                path = f'packs/emotion-{index:02d}.json'
                for member in pack:
                    assert member not in locations
                    locations[member] = path
                write_atomic(STAGE / path, (json.dumps(pack, separators=(',', ':')) + '\n').encode())
                groups.append({'emotion': by_emotion[index][0]['emotion'],
                               'cases': public_cases})
        finally:
            for archive in archives:
                archive.close()
    assert len(locations) == 160 * (1 + 2 * len(ARMS))
    data = {'schema': 'chatterbox-s3gen-vc-grow-24k-listening-v1',
            'targets': [{key: target[key] for key in ('uid', 'label', 'lang', 'text', 'dur_s')}
                        for target in site['targets']],
            'references': references, 'groups': groups, 'audio_locations': locations,
            'metrics': metric_js, 'arms': list(ARMS), 'labels': LABELS,
            'summary': {'selected_steps': done['selected_steps'],
                        'test_metrics': test_metrics, 'selection_cases': 40,
                        'unseen_test_cases': 83, 'training_overlap_cases': 37}}
    compact = json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c')
    template = (HERE / 'space_template.html').read_text()
    assert template.count('__DATA_JSON__') == 1
    html = template.replace('__DATA_JSON__', compact)
    write_atomic(STAGE / 'index.html', html.encode('utf-8'))
    readme = f'''---
title: Chatterbox S3Gen VC · 24k RL Listening
emoji: 🎧
colorFrom: blue
colorTo: purple
sdk: static
app_file: index.html
license: cc-by-4.0
---

# Chatterbox S3Gen VC · 24k RL listening comparison

Static, lazy-loaded listening comparison of the 300-hour supervised baseline
and three RL follow-up adapters at their selected checkpoints. The 160 source
clips and two fixed target voices come from the
[original S10 listening room](https://adopt-judge-jokes-sea.trycloudflare.com/).
The original user-provided
source may be temporary; the source UIDs, audio SHA-256 values and target
identifiers are frozen in `INPUTS.json`. All 1,280 prediction MP3s are packed
into forty emotion-level JSON containers; there are no per-sample MP3 files.

Selection used 40 RL-training-UID-disjoint cases, two targets and the same
seed per case/model. The main metrics table uses 83 other UID-disjoint cases,
excluding 37 known source-UID overlaps with RL training. This is **not** a
guarantee of speaker-level disjointness. Content Enjoyment and emotion scores
are produced by the same Humaneness Ears Medium model used in training, so
they are not independent quality evidence. CAMPPlus cosine is a proxy for
speaker similarity. No human MOS or WER claim is made.

Selected steps: {json.dumps(done['selected_steps'], sort_keys=True)}.

Models: [Continue](https://huggingface.co/laion/chatterbox-s3gen-vc-grow-ce-continue),
[Re-merged](https://huggingface.co/laion/chatterbox-s3gen-vc-grow-ce-remerged),
[Speaker-Mix](https://huggingface.co/laion/chatterbox-s3gen-vc-grow-ce-speaker),
[supervised parent](https://huggingface.co/laion/chatterbox-s3gen-vc-grow).

LAION-created UI and evaluation metadata are CC BY 4.0. Source audio retains
its individual upstream licensing; see `INPUTS.json`. Chatterbox retains its
separate upstream license.
'''
    write_atomic(STAGE / 'README.md', readme.encode('utf-8'))
    input_manifest = {'schema': site['schema'], 'source_page': site['source_page'],
                      'targets': [{key: target[key] for key in
                                   ('uid', 'label', 'lang', 'text', 'license', 'hf_source')}
                                  for target in site['targets']],
                      'cases': [{key: case[key] for key in
                                 ('uid', 'case_id', 'lang', 'origin', 'text', 'emotion',
                                  'license', 'hf_source')}
                                for case in cases],
                      'source_audio_sha256': site['audio_sha256'],
                      'source_tar_sha256': site['tar_sha256']}
    write_atomic(STAGE / 'INPUTS.json',
                 (json.dumps(input_manifest, indent=2, ensure_ascii=False) + '\n').encode())
    report = {'schema': data['schema'], 'repo': REPO, 'selected_steps': done['selected_steps'],
              'test_metrics': test_metrics, 'selection': selection,
              'sources': 160, 'targets': 2, 'predictions': len(metrics),
              'audio_packs': 40}
    write_atomic(STAGE / 'RESULTS.json',
                 (json.dumps(report, indent=2) + '\n').encode())
    proof = {'status': 'COMPLETE', 'repo': REPO, 'packs': 40,
             'predictions': len(metrics), 'index_sha256': hashlib.sha256(
                 (STAGE / 'index.html').read_bytes()).hexdigest()}
    write_atomic(OUT / 'SPACE_BUILD_DONE.json',
                 (json.dumps(proof, indent=2) + '\n').encode())
    print(json.dumps(proof), flush=True)


def remote_files(api: HfApi) -> dict[str, tuple[str | None, int | None]]:
    info = api.repo_info(repo_id=REPO, repo_type='space', files_metadata=True)
    return {item.rfilename: (item.lfs.sha256 if item.lfs else None, item.size)
            for item in info.siblings}


def publish() -> None:
    api = HfApi()
    api.create_repo(repo_id=REPO, repo_type='space', space_sdk='static',
                    private=False, exist_ok=True)
    state = read_json(STATE) if STATE.exists() else {'files': {}}
    # Publish the page last, so visitors never see broken audio links.
    paths = [STAGE / 'README.md', STAGE / 'INPUTS.json', STAGE / 'RESULTS.json']
    paths += sorted((STAGE / 'packs').glob('*.json'))
    paths.append(STAGE / 'index.html')
    assert len(paths) == 44
    remote = remote_files(api)
    for source in paths:
        name = str(source.relative_to(STAGE))
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        size = source.stat().st_size
        if state['files'].get(name) == {'sha256': digest, 'bytes': size}:
            if name in remote and remote[name][1] == size:
                continue
        api.upload_file(repo_id=REPO, repo_type='space', path_or_fileobj=str(source),
                        path_in_repo=name, commit_message=f'Publish {name}')
        remote = remote_files(api)
        if name not in remote or remote[name][1] != size:
            raise RuntimeError(f'Remote size verification failed: {name}')
        if remote[name][0] is not None and remote[name][0] != digest:
            raise RuntimeError(f'Remote LFS SHA-256 verification failed: {name}')
        state['files'][name] = {'sha256': digest, 'bytes': size}
        write_atomic(STATE, (json.dumps(state, indent=2) + '\n').encode())
        print(json.dumps({'uploaded': name, 'bytes': size}), flush=True)
    assert len(state['files']) >= 44
    write_atomic(OUT / 'SPACE_PUBLISHED.json',
                 (json.dumps({'status': 'COMPLETE', 'repo': REPO,
                              'url': f'https://{REPO.replace("/", "-")}.static.hf.space/',
                              'files': 44}, indent=2) + '\n').encode())


def main() -> None:
    if not (OUT / 'SPACE_BUILD_DONE.json').exists():
        build()
    publish()


if __name__ == '__main__':
    main()
