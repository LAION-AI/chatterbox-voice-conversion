#!/usr/bin/env python3
"""Add the 40k full-flow arm to the existing LAION curated VC Space page.

Runs on a login node after EVAL_DONE. The same 40 source clips, A/B targets,
seeds, old audio packs and old metrics are retained. Eight new JSON audio
containers are uploaded before the page, so no visible player points to a
missing asset. The all-case/UID-disjoint proxy comparison is shown above the
listening grid. No loose per-sample audio files are written.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import tarfile
import time
import urllib.request

from huggingface_hub import CommitOperationAdd, HfApi
import numpy as np
import pyarrow.parquet as pq

OUT = Path('/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_v4_full_403030')
EVAL = OUT / 'evaluation'
OLD = Path('/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_followup_eval')
STAGE = OLD / 'space_stage'
REPO = 'laion/chatterbox-s3gen-vc-grow-24k-listening'
URL = 'https://laion-chatterbox-s3gen-vc-grow-24k-listening.static.hf.space/curated.html'
START = '<script id="eval-data" type="application/json">'
END = '</script>'


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_bytes(data)
    os.replace(temporary, path)


def metric_means(rows: list[dict]) -> dict:
    assert rows
    return {key: float(np.mean([row[key] for row in rows]))
            for key in ('audiobox_ce', 'audiobox_pq',
                        'top2_emotion_mse', 'target_camp_cosine')}


def table_row(label: str, metrics: dict) -> str:
    return ('<tr><td>' + label + '</td>' + ''.join(
        f'<td>{metrics[key]:.3f}</td>' for key in
        ('audiobox_ce', 'audiobox_pq', 'top2_emotion_mse',
         'target_camp_cosine')) + '</tr>')


def build() -> dict:
    done = read_json(EVAL / 'EVAL_DONE.json')
    assert done['status'] == 'COMPLETE' and done['step'] == 40000
    assert done['sources'] == 160 and done['predictions'] == 320
    assert read_json(OUT / 'DONE.json')['step'] == 40000
    original = STAGE / 'curated-before-full40k.html'
    if not original.exists():
        old_page = (STAGE / 'curated.html').read_bytes()
        assert b'full40k' not in old_page
        atomic(original, old_page)
    page = original.read_text()
    # The HTML also has its executable UI script, so only the data-script
    # opener is unique; split at the first following closing script tag.
    assert page.count(START) == 1
    left, rest = page.split(START, 1)
    payload, right = rest.split(END, 1)
    data = json.loads(payload)
    assert len(data['cases']) == 40 and len(data['targets']) == 2
    assert 'full40k' not in data['arms']
    selected = {case['case_id'] for case in data['cases']}
    assert selected == set(read_json(STAGE / 'CURATED.json')['selected_case_ids'])
    rows = pq.read_table(EVAL / 'metrics.parquet').to_pylist()
    assert len(rows) == 320
    metric_map = {(r['case_id'], r['target_label']): r for r in rows}
    assert len(metric_map) == 320
    needed = {f'{cid}/full40k_vc_{label}.mp3'
              for cid in selected for label in ('A', 'B')}
    audio = {}
    for rank in range(4):
        with tarfile.open(EVAL / f'audio-rank{rank}.tar', 'r') as archive:
            for name in needed - audio.keys():
                try:
                    member = archive.extractfile(name)
                except KeyError:
                    continue
                assert member is not None
                audio[name] = member.read()
    assert set(audio) == needed and all(audio.values())
    pack_paths = []
    for group in range(8):
        members = {name: base64.b64encode(blob).decode('ascii')
                   for name, blob in audio.items()
                   if int(name.split('-', 1)[0]) // 5 == group}
        assert len(members) == 10
        path = STAGE / f'packs/full40k-{group:02d}.json'
        atomic(path, (json.dumps(members, separators=(',', ':')) + '\n').encode())
        pack_paths.append(path)
    data['arms'].append('full40k')
    data['labels']['full40k'] = 'Full-flow 40k · CE/emotion/identity'
    data['selected_steps']['full40k'] = 40000
    curated = [row for row in rows if row['case_id'] in selected]
    assert len(curated) == 80
    data['summary']['full40k'] = metric_means(curated)
    for row in curated:
        member = row['mp3_member']
        assert member in audio and member not in data['locations']
        group = int(member.split('-', 1)[0]) // 5
        data['locations'][member] = f'packs/full40k-{group:02d}.json'
        data['metrics'][member] = {key: row[key] for key in
                                   ('audiobox_ce', 'audiobox_pq',
                                    'top2_emotion_mse', 'target_camp_cosine')}
    data['schema'] = 'chatterbox-s3gen-vc-curated-full40k-v2'
    old_metrics = done['matched_24k_on_uid_disjoint_noncurated']
    holdout_n = done['uid_disjoint_noncurated_case_count']
    assert holdout_n == 93
    new_metrics = done['metrics']['uid_disjoint_noncurated']
    disjoint_table = (
        '<section class="card stats"><h2>UID-disjoint comparison · 93 other sources</h2>'
        '<p class="note">These 93 sources were not used as known RL training UIDs '
        'and are separate from the 40 curated listening sources. All arms use the '
        'same 93 inputs, two targets and seeds. Speaker-level disjointness is not proven. '
        'CE, PQ and emotion are Humaneness Ears Medium proxy scores (the reward '
        'model), not independent human listening ratings. CAMPPlus is a proxy '
        'for target voice similarity.</p><table><thead><tr><th>Model</th>'
        '<th>CE ↑</th><th>PQ ↑</th><th>Emotion MSE ↓</th>'
        '<th>Speaker cosine ↑</th></tr></thead><tbody>'
        + table_row('300h supervised baseline', old_metrics['baseline'])
        + table_row('Speaker-Mix 24k LoRA', old_metrics['speaker_mix'])
        + table_row('Full-flow 40k', new_metrics)
        + '</tbody></table></section>')
    left = left.replace('three 24k-group RL models against',
                        'three 24k-group RL models and their 40k full-flow continuation against')
    left = left.replace('four tuned columns', 'five tuned columns')
    left = left.replace('repeat(6,minmax(135px,1fr))',
                        'repeat(7,minmax(135px,1fr))')
    assert 'five tuned columns' in left and 'repeat(7,minmax(135px,1fr))' in left
    anchor = '<details class="method">'
    assert left.count(anchor) == 1
    left = left.replace(anchor, disjoint_table + anchor)
    hero = '<h1>English-first listening room: from vanilla to RL</h1>'
    assert left.count(hero) == 1
    left = left.replace(hero, hero +
        '<p class="note"><b>New:</b> the 40k arm continues the 24k Speaker-Mix '
        'LoRA after merging it into the Chatterbox flow decoder. It trains all '
        '112.6M flow-generator parameters (the upstream tokenizer, CAMPPlus and '
        'HiFT remain frozen), in groups of 16 on four GPUs. Its within-group '
        'reward weights are 40% AudioBox Content Enjoyment from Humaneness Ears '
        'Medium, 30% negative source-top-two-emotion MSE and 30% target CAMPPlus '
        'cosine. The single warm-up/cosine schedule spans all 40,000 groups.</p>')
    compact = json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c')
    rendered = left + START + compact + END + right
    assert rendered.count('full40k') >= 80
    atomic(STAGE / 'curated.html', rendered.encode())
    public = {'schema': data['schema'], 'status': 'COMPLETE', 'step': 40000,
              'sources': 160, 'targets': 2, 'predictions': 320,
              'curated_cases': 40, 'uid_disjoint_noncurated_cases': holdout_n,
              'curated_metrics': data['summary']['full40k'],
              'uid_disjoint_metrics': new_metrics,
              'comparison': {arm: old_metrics[arm]
                             for arm in ('baseline', 'speaker_mix')},
              'limitations': done['limitations'],
              'page': 'curated.html', 'audio_pack_count': len(pack_paths),
              'page_sha256': hashlib.sha256(rendered.encode()).hexdigest()}
    atomic(STAGE / 'FULL40K_EVAL.json',
           (json.dumps(public, indent=2, ensure_ascii=False) + '\n').encode())
    readme = STAGE / 'README.md'
    content = readme.read_text()
    marker = '## Full-flow 40k continuation'
    if marker not in content:
        content += ('\n' + marker + '\n\n'
                    'The [curated listening room](curated.html) now includes a '
                    'matched-seed full-flow 40,000-group continuation of the '
                    '24k Speaker-Mix adapter. It uses the same 40 curated inputs '
                    'and two fixed target voices; 320 new predictions over all '
                    '160 inputs were scored. [FULL40K_EVAL.json](FULL40K_EVAL.json) '
                    'contains the 93 other training-UID-disjoint-case comparison, '
                    'reward details and limitations. New MP3s are stored in eight '
                    'packed JSON audio containers, not loose files.\n')
        atomic(readme, content.encode())
    build_proof = {'status': 'COMPLETE', 'step': 40000,
                   'audio_packs': len(pack_paths), 'predictions': 320,
                   'curated_predictions': 80,
                   'page_sha256': public['page_sha256']}
    atomic(EVAL / 'SPACE_BUILD_DONE.json',
           (json.dumps(build_proof, indent=2) + '\n').encode())
    return build_proof


def remote_files(api: HfApi) -> tuple[str, dict]:
    info = api.repo_info(repo_id=REPO, repo_type='space', files_metadata=True)
    return info.sha, {item.rfilename: item for item in info.siblings}


def verify(api: HfApi, paths: list[Path]) -> str:
    revision, remote = remote_files(api)
    for path in paths:
        name = str(path.relative_to(STAGE))
        assert name in remote and remote[name].size == path.stat().st_size, name
        if remote[name].lfs:
            assert remote[name].lfs.sha256 == hashlib.sha256(path.read_bytes()).hexdigest(), name
    return revision


def publish() -> dict:
    if (EVAL / 'SPACE_PUBLISHED.json').exists():
        published = read_json(EVAL / 'SPACE_PUBLISHED.json')
        assert published['status'] == 'COMPLETE'
        return published
    if not (EVAL / 'SPACE_BUILD_DONE.json').exists():
        build()
    packs = [STAGE / f'packs/full40k-{group:02d}.json' for group in range(8)]
    metadata = [STAGE / 'FULL40K_EVAL.json', STAGE / 'README.md',
                STAGE / 'curated.html']
    assert all(path.exists() for path in packs + metadata)
    api = HfApi()
    api.create_commit(repo_id=REPO, repo_type='space',
                      operations=[CommitOperationAdd(path_in_repo=str(path.relative_to(STAGE)),
                                                     path_or_fileobj=str(path))
                                  for path in packs],
                      commit_message='Add packed full-flow 40k comparison audio')
    verify(api, packs)
    api.create_commit(repo_id=REPO, repo_type='space',
                      operations=[CommitOperationAdd(path_in_repo=str(path.relative_to(STAGE)),
                                                     path_or_fileobj=str(path))
                                  for path in metadata],
                      commit_message='Add full-flow 40k metrics and curated Space column')
    revision = verify(api, packs + metadata)
    # Static Space deployment is asynchronous. Only mark the job finished once
    # the served page actually contains the new arm.
    for attempt in range(18):
        try:
            with urllib.request.urlopen(URL + '?v=' + revision[:10], timeout=20) as response:
                live = response.read().decode('utf-8')
            if 'Full-flow 40k' in live and 'packs/full40k-00.json' in live:
                break
        except Exception as error:
            print(json.dumps({'event': 'SPACE_WAIT', 'attempt': attempt + 1,
                              'error': repr(error)}), flush=True)
        time.sleep(20)
    else:
        raise RuntimeError('HF files are verified, but the static Space has not refreshed')
    proof = {'status': 'COMPLETE', 'revision': revision, 'url': URL,
             'audio_packs': len(packs), 'step': 40000}
    atomic(EVAL / 'SPACE_PUBLISHED.json',
           (json.dumps(proof, indent=2) + '\n').encode())
    print(json.dumps(proof), flush=True)
    return proof


if __name__ == '__main__':
    publish()
