#!/usr/bin/env python3
"""Build/publish curated VC demo v2 from packed, already scored source material.

No per-sample local files are created. New external and Best-of-N predictions
come from independently completed, rank-wise TAR/Parquet Slurm evaluations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tarfile

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
import pyarrow.parquet as pq

ROOT = Path('/e/scratch/laionize/schuhmann1')
INPUT = ROOT / 'chatterbox_s3gen_grow_followup_eval'
VANILLA = INPUT / 'vanilla_curation'
FULL = ROOT / 'chatterbox_s3gen_grow_v4_full_403030/evaluation'
WORK = ROOT / 'hvc_demo_v2'
STAGE = WORK / 'stage'
sys.path.insert(0, '/e/home/jusers/schuhmann1/jupiter/chatterbox_s3gen_300h_v1/hvc_release')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_demo_space import ARMS, LABELS, FIELDS, AUDIO_REPO, SPACE_REPO, MODEL_REPO, page_html, space_readme


def packed_audio(directory: Path, prefix: str, selected: set[str]) -> dict[str, bytes]:
    found = {}
    for rank in range(4):
        with tarfile.open(directory / f'{prefix}-rank{rank}.tar', 'r') as archive:
            for member in archive:
                if not member.isfile() or member.name.split('/', 1)[0] not in selected:
                    continue
                assert member.name not in found and member.name.endswith('.mp3')
                with archive.extractfile(member) as handle:
                    found[member.name] = handle.read()
    return found


def mean(rows: list[dict]) -> dict:
    return {key: sum(float(row[key]) for row in rows) / len(rows) for key in FIELDS}


def build() -> dict[str, bytes]:
    curation = json.loads((WORK / 'CURATION.json').read_text())
    selected = set(curation['selected_case_ids'])
    assert len(selected) == 40
    inventory = {row['case_id']: row for row in
                 json.loads((INPUT / 'space_stage/INPUTS.json').read_text())['cases']}
    prepared = {row['case_id']: row for row in pq.read_table(
        INPUT / 'prepared.parquet', columns=['case_id', 'role', 'rl_training_uid_overlap']).to_pylist()
        if row['role'] == 'source'}
    vanilla_rows = {}
    for rank in range(4):
        for line in (VANILLA / f'vanilla-rank{rank}.jsonl').read_text().splitlines():
            row = json.loads(line)
            vanilla_rows[row['case_id']] = row
    assert len(vanilla_rows) == 160
    full_rows = {(row['case_id'], row['target_label']): row for row in
                 pq.read_table(FULL / 'metrics.parquet').to_pylist()}
    assert len(full_rows) == 320
    ext_rows = {arm: {(row['case_id'], row['target_label']): row for row in
                      pq.read_table(WORK / 'evaluation' / arm / 'full/metrics.parquet').to_pylist()}
                for arm in ('seed_v1', 'seed_v2', 'xvc')}
    assert all(len(rows) == 80 for rows in ext_rows.values())
    bon_rows = pq.read_table(WORK / 'bon/full/all_winners.parquet').to_pylist()
    assert len(bon_rows) == 320
    bon = {(row['case_id'], row['target_label'], row['pool_size']): row for row in bon_rows}
    assert len(bon) == 320
    done = json.loads((WORK / 'bon/full/DONE.json').read_text())
    assert done['status'] == 'COMPLETE'

    sources = {}
    with tarfile.open(INPUT / 'site_inputs.tar', 'r') as archive:
        for cid in selected:
            with archive.extractfile(f'sources/{cid}.mp3') as handle:
                sources[f'{cid}/original.mp3'] = handle.read()
        references = {}
        for label in ('A', 'B'):
            with archive.extractfile(f'references/{label}.mp3') as handle:
                references[label] = handle.read()
    packs = {
        'vanilla': packed_audio(VANILLA, 'vanilla', selected),
        'full40k': packed_audio(FULL, 'audio', selected),
        'bon': packed_audio(WORK / 'bon/full', 'winners', selected),
    }
    for arm in ('seed_v1', 'seed_v2', 'xvc'):
        packs[arm] = packed_audio(WORK / 'evaluation' / arm / 'full', 'audio', selected)

    ordered = sorted(selected, key=lambda cid: (
        inventory[cid]['origin'] != 'real', inventory[cid]['lang'] != 'en',
        inventory[cid]['emotion'], cid))
    audio = {}
    cases = []
    metric_rows = {arm: [] for arm in ARMS}
    for cid in ordered:
        meta = inventory[cid]
        assert meta['license'] == 'CC-BY-4.0'
        case = {key: meta[key] for key in ('case_id', 'uid', 'emotion', 'lang', 'origin',
                                           'text', 'license', 'hf_source')}
        case['rl_training_uid_overlap'] = bool(prepared[cid]['rl_training_uid_overlap'])
        case['source_quality'] = {key: vanilla_rows[cid][key] for key in (
            'source_audiobox_ce', 'source_audiobox_pq',
            'source_predicted_p808_mos', 'source_camp_window_min_cosine')}
        case['audio'] = {'original': f'audio/{cid}/original.mp3'}
        audio[case['audio']['original']] = sources[f'{cid}/original.mp3']
        case['arms'] = {}
        for arm in ARMS:
            case['arms'][arm] = {}
            for label in ('A', 'B'):
                if arm == 'vanilla':
                    row = vanilla_rows[cid]['predictions'][label]
                    member, pack = f'{cid}/vanilla_vc_{label}.mp3', 'vanilla'
                elif arm == 'full40k':
                    row = full_rows[(cid, label)]
                    member, pack = f'{cid}/full40k_vc_{label}.mp3', 'full40k'
                elif arm.startswith('bon'):
                    row = bon[(cid, label, int(arm[3:]))]
                    member, pack = f'{cid}/{arm}_vc_{label}.mp3', 'bon'
                else:
                    row = ext_rows[arm][(cid, label)]
                    member, pack = f'{cid}/{arm}_vc_{label}.mp3', arm
                path = f'audio/{member}'
                audio[path] = packs[pack][member]
                case['arms'][arm][label] = {'path': path,
                    'metrics': {field: float(row[field]) for field in FIELDS}}
                metric_rows[arm].append(row)
        cases.append(case)
    targets = pq.read_table(INPUT / 'prepared.parquet').to_pylist()
    target_map = {}
    for row in targets:
        if row['role'] == 'target':
            label = row['target_label']
            path = f'audio/references/{label}.mp3'
            audio[path] = references[label]
            target_map[label] = {'path': path, 'text': row['text'],
                                 'lang': row['lang'], 'uid': row['uid']}
    assert len(target_map) == 2 and len(audio) == 682
    assert all(payload[:3] == b'ID3' for payload in audio.values())
    manifest = {
        'schema': 'humaneness-vc-standalone-demo-v2',
        'model_repo': MODEL_REPO, 'audio_repo': AUDIO_REPO,
        'cases': cases, 'references': target_map,
        'arms': ARMS, 'labels': LABELS,
        'summary': {arm: mean(rows) for arm, rows in metric_rows.items()},
        'best_of_n': {str(n): done['summary'][str(n)] for n in (1, 4, 8, 16)},
        'audio_count': len(audio),
        'method': {'cases': 40, 'targets': 2,
                   'english_cases': sum(c['lang'] == 'en' for c in cases),
                   'german_cases': sum(c['lang'] == 'de' for c in cases),
                   'real_cases': sum(c['origin'] == 'real' for c in cases),
                   'distinct_emotions': curation['distinct_emotions'],
                   'rl_training_uid_overlap_cases': sum(c['rl_training_uid_overlap'] for c in cases),
                   'selection': curation['selection_policy'],
                   'model': 'Humaneness Voice Converter: 40k full-flow after Speaker-Mix',
                   'proxy_judge': 'Humaneness Ears Medium for CE/PQ/emotions; CAMPPlus for target cosine',
                   'caveat': 'Not blinded subjective, independent judge, WER or speaker-disjoint study.'}}
    STAGE.mkdir(parents=True, exist_ok=True)
    (STAGE / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    (STAGE / 'index.html').write_text(page_html(manifest))
    (STAGE / 'README.md').write_text(space_readme())
    (STAGE / 'AUDIO_README.md').write_text(f'''---
license: cc-by-4.0
pretty_name: Humaneness Voice Converter Demo Audio
tags:
  - audio
  - voice-conversion
---

# Humaneness Voice Converter Demo Audio

682 individual 24-kHz mono 128-kb/s MP3s for the [listening demo](https://huggingface.co/spaces/{SPACE_REPO}): 40 sources ({manifest['method']['real_cases']} real, {manifest['method']['german_cases']} German), two target references and eight converter/selection arms per source and target. The public Space manifest has source text, attribution, provenance and per-case metrics. The curation selects cleaner, more coherent material from 160 licensed candidates without using model-output scores.

Source clips are CC BY 4.0 in the input inventory; target A is CC0 and B is CC BY 4.0. Outputs retain source rights. External converters retain their own licenses and are credited in the Space. Use only with speaker/rightsholder authorization.
''')
    print(json.dumps({'event': 'BUILT', 'cases': len(cases),
                      'real': manifest['method']['real_cases'],
                      'german': manifest['method']['german_cases'],
                      'audio': len(audio), 'bytes': sum(map(len, audio.values())),
                      'summary': manifest['summary']}), flush=True)
    return audio


def publish(audio: dict[str, bytes]) -> None:
    api = HfApi()
    assert any(org['name'] == 'laion' for org in api.whoami().get('orgs', []))
    api.create_repo(AUDIO_REPO, repo_type='dataset', exist_ok=True)
    current = set(api.list_repo_files(AUDIO_REPO, repo_type='dataset'))
    missing = [(p, b) for p, b in sorted(audio.items()) if p not in current]
    print(json.dumps({'event': 'UPLOAD_AUDIO', 'missing': len(missing)}), flush=True)
    for start in range(0, len(missing), 20):
        api.create_commit(repo_id=AUDIO_REPO, repo_type='dataset',
                          operations=[CommitOperationAdd(path_in_repo=p, path_or_fileobj=b)
                                      for p, b in missing[start:start + 20]],
                          num_threads=1, commit_message='Refresh curated converter listening clips')
        print(json.dumps({'event': 'AUDIO_PROGRESS', 'done': min(start + 20, len(missing)),
                          'total': len(missing)}), flush=True)
    api.upload_file(path_or_fileobj=str(STAGE / 'AUDIO_README.md'), path_in_repo='README.md',
                    repo_id=AUDIO_REPO, repo_type='dataset',
                    commit_message='Document refreshed real-speech-heavy demo')
    assert set(audio).issubset(set(api.list_repo_files(AUDIO_REPO, repo_type='dataset')))
    api.create_repo(SPACE_REPO, repo_type='space', space_sdk='static', exist_ok=True)
    for name in ('README.md', 'manifest.json', 'index.html'):
        api.upload_file(path_or_fileobj=str(STAGE / name), path_in_repo=name,
                        repo_id=SPACE_REPO, repo_type='space',
                        commit_message=f'Refresh converter demo {name}')
    # Removed examples stay in the local packed archive/HF revision history,
    # but are removed from the current public audio tree after the new Space
    # and all its referenced MP3s have landed.
    stale = sorted(p for p in current if p.startswith('audio/') and p not in audio)
    for start in range(0, len(stale), 50):
        api.create_commit(repo_id=AUDIO_REPO, repo_type='dataset',
                          operations=[CommitOperationDelete(path_in_repo=p)
                                      for p in stale[start:start + 50]],
                          num_threads=1, commit_message='Remove superseded demo audio clips')
    print(json.dumps({'event': 'PUBLISHED', 'space': f'https://huggingface.co/spaces/{SPACE_REPO}',
                      'new_audio': len(missing), 'retired_audio': len(stale)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--build-only', action='store_true')
    args = parser.parse_args()
    payload = build()
    if not args.build_only:
        publish(payload)
