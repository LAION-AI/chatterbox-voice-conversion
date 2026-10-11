#!/usr/bin/env python3
"""Publish the three CE follow-ups and each COMPLETE checkpoint to LAION HF.

Run this on the login node. Only complete, size-matching checkpoint files are
eligible; the remote LFS SHA-256 and size are checked before indexing them.
Training stays on Scratch and this process never stages checkpoint copies in
Home, avoiding extra inode use and temporary multi-GB files.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import time

from huggingface_hub import HfApi

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
BASE = PROJECT / 'hf_release'
V1 = PROJECT / 'grow_v1'
V2 = PROJECT / 'grow_v2'
V3 = PROJECT / 'grow_v3'
STATE_PATH = HERE / 'HF_FOLLOWUP_UPLOAD_STATE.json'
LOG_PATH = HERE / 'HF_FOLLOWUP_UPLOAD_LOG.jsonl'
SPEAKER_DEFAULT = '''## Recommended checkpoint and listening evidence

**Default recommendation for this RL release:**
[`checkpoints/step-0024000.pt`](checkpoints/step-0024000.pt). In the 40-source-UID-disjoint
checkpoint-selection panel, step 24,000 had the best predefined combined
Content Enjoyment / emotion / identity score among steps 8k, 16k and 24k.
Use the included inference code with `--arm speaker_mix`; the checkpoint is
an adapter plus optimizer state, **not** a standalone Chatterbox model. This
recommendation selects a starting point *among these RL adapters*, not a
claim that it beats the supervised or vanilla model on every dimension.

Listen to the [curated 40-emotion comparison](https://laion-chatterbox-s3gen-vc-grow-24k-listening.static.hf.space/curated.html)
with the exact same source clips, target voices and seeds across Vanilla,
supervised SFT and all three 24k RL arms. The table below shows that page's
means (higher CE/PQ/speaker cosine and lower emotion MSE are directionally
better):

| Model | Content Enjoyment ↑ | AudioBox PQ ↑ | Top-2 emotion MSE ↓ | Target CAMPPlus cosine ↑ |
|---|---:|---:|---:|---:|
| Vanilla Chatterbox | 6.173 | 7.747 | 0.140 | **0.806** |
| Supervised 300h | 6.151 | 7.695 | 0.113 | 0.778 |
| RL Continue, 24k | **6.384** | **7.957** | **0.111** | 0.714 |
| RL Re-merged, 24k | 6.370 | 7.950 | 0.113 | 0.725 |
| **RL Speaker-Mix, 24k** | 6.348 | 7.928 | 0.119 | **0.752** |

The curated page has 32 English and eight German clips, one per emotion;
10 source UIDs overlap RL training. Selection used predicted P808 MOS and a
CAMPPlus within-clip consistency proxy, **not** human MOS or diarization. It
is a listening preview, not an independent held-out benchmark; CE and emotion
are scored by the same frozen Ears model used in the RL reward. On the
[separate 83-case source-UID-disjoint comparison](https://laion-chatterbox-s3gen-vc-grow-24k-listening.static.hf.space/),
Speaker-Mix improves CE over supervised SFT (6.031 vs 5.766), but trails it
in target-speaker cosine (0.737 vs 0.761) and top-two emotion MSE (0.252 vs
0.228). In that test Speaker-Mix still preserves speaker cosine better than
the other two RL arms. No WER or human MOS was measured here.
'''
RUNS = {
    'continue': {
        'repo': 'laion/chatterbox-s3gen-vc-grow-ce-continue',
        'root': Path('/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_v2_ce'),
        'title': 'Continued rank-128 LoRA, 50/50 CE and emotion',
        'arm_description': 'The completed v1 `continue` rank-128 adapter at step 4,000 is continued without merging it into the base.',
        'reward_sentence': 'The weights are **50% Content Enjoyment and 50% negative top-two emotion MSE**.',
        'extra_download': '',
        'extra_flag': '',
        'reconstruction_note': 'Load the supervised base adapter, then load this continuation adapter.',
        'default_recommendation': '',
        'default_checkpoint': 'step-0024000.pt',
    },
    'remerged_new': {
        'repo': 'laion/chatterbox-s3gen-vc-grow-ce-remerged',
        'root': Path('/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_v2_ce'),
        'title': 'Re-merged fresh rank-128 LoRA, 50/50 CE and emotion',
        'arm_description': 'The completed v1 `merged_new` step-4,000 adapter is merged into its effective base and a new rank-128 adapter is initialized for this follow-up.',
        'reward_sentence': 'The weights are **50% Content Enjoyment and 50% negative top-two emotion MSE**.',
        'extra_download': 'hf download laion/chatterbox-s3gen-vc-grow \\\n+  checkpoints/grow_merged_new_step0004000_adapter.safetensors \\\n+  --local-dir parent_v1',
        'extra_flag': '--parent-v1 parent_v1/checkpoints/grow_merged_new_step0004000_adapter.safetensors \\\n+  ',
        'reconstruction_note': 'Rebuild the v1 merged-new effective base before applying this fresh adapter; the supplied inference code does that.',
        'default_recommendation': '',
        'default_checkpoint': 'step-0024000.pt',
    },
    'speaker_mix': {
        'repo': 'laion/chatterbox-s3gen-vc-grow-ce-speaker',
        'root': Path('/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_v3_ce_emo_speaker'),
        'title': 'Continued rank-128 LoRA, 40/40/20 CE, emotion and speaker',
        'arm_description': 'The completed v1 `continue` rank-128 adapter at step 4,000 is continued, now with an explicit target-speaker reward.',
        'reward_sentence': 'The weights are **40% Content Enjoyment, 40% negative top-two emotion MSE, and 20% target CAMPPlus speaker cosine**.',
        'extra_download': '',
        'extra_flag': '',
        'reconstruction_note': 'Load the supervised base adapter, then load this continuation adapter.',
        'default_recommendation': SPEAKER_DEFAULT,
        'default_checkpoint': 'step-0024000.pt',
    },
}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def emit(event: str, **kw) -> None:
    row = {'utc': utc(), 'event': event, **kw}
    line = json.dumps(row, sort_keys=True)
    print(line, flush=True)
    with LOG_PATH.open('a') as handle:
        handle.write(line + '\n')


def save_state(state: dict) -> None:
    temporary = STATE_PATH.with_suffix('.json.partial')
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, STATE_PATH)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def public_files(arm: str, config: dict) -> dict[str, Path | bytes]:
    template = (HERE / 'README_TEMPLATE.md').read_text()
    readme = template.format(arm=arm, **{k: v for k, v in config.items()
                                        if k not in ('repo', 'root')}).encode()
    files: dict[str, Path | bytes] = {
        'README.md': readme,
        'TRAINING.md': HERE / 'TRAINING.md',
        'requirements.txt': HERE / 'requirements.txt',
        'score_order_and_preparation.json': BASE / 'score_order_and_preparation.json',
        'code/grow_followup/infer_followup.py': HERE / 'infer_followup.py',
        'code/grow_followup/publish.py': HERE / 'publish.py',
        'code/grow_v2/train_ddp.py': V2 / 'train_ddp.py',
        'code/grow_v2/run.sbatch': V2 / 'run.sbatch',
        'code/grow_v3/train_speaker.py': V3 / 'train_speaker.py',
        'code/grow_v3/run.sbatch': V3 / 'run.sbatch',
    }
    for name in ('train.py', 'generate_listening.py', 'select_listening.py',
                 'prepare.py', 's3_tokenization.py'):
        files[f'code/{name}'] = BASE / 'code' / name
    for name in ('train_grow.py', 'score_inputs.py', 'choose_pairs.py',
                 'prepare_pool.py', 'infer_packed.py'):
        files[f'code/grow_v1/{name}'] = V1 / name
    return files


def remote_files(api: HfApi, repo: str) -> dict[str, tuple[str | None, int | None]]:
    info = api.model_info(repo, files_metadata=True)
    return {f.rfilename: (f.lfs.sha256 if f.lfs else None, f.size)
            for f in info.siblings}


def bootstrap(api: HfApi, arm: str, config: dict, state: dict) -> None:
    repo = config['repo']
    api.create_repo(repo_id=repo, repo_type='model', private=False, exist_ok=True)
    docs = state.setdefault('docs', {}).setdefault(arm, {})
    for path, source in public_files(arm, config).items():
        if isinstance(source, bytes):
            payload = source
            data = io.BytesIO(payload)
        else:
            payload = source.read_bytes()
            data = str(source)
        digest = hashlib.sha256(payload).hexdigest()
        if docs.get(path) == digest:
            continue
        api.upload_file(repo_id=repo, repo_type='model', path_or_fileobj=data,
                        path_in_repo=path, commit_message=f'Update {path}')
        docs[path] = digest
        save_state(state)
        emit('DOC_UPLOADED', arm=arm, path=path, sha256=digest)


def candidates(arm: str, config: dict) -> list[tuple[int, Path, str]]:
    directory = config['root'] / arm / 'checkpoints'
    found = []
    for marker in directory.glob('step-*.COMPLETE.json'):
        try:
            proof = json.loads(marker.read_text())
            step = int(marker.name.split('.')[0].split('-')[1])
            file = directory / f'step-{step:07d}.pt'
            if (proof.get('status') != 'COMPLETE' or proof.get('arm') != arm
                    or proof.get('step') != step or not file.is_file()
                    or file.stat().st_size != proof.get('bytes')):
                emit('SKIP_INCOMPLETE', arm=arm, marker=str(marker))
                continue
            found.append((step, file, f'checkpoints/step-{step:07d}.pt'))
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            emit('SKIP_BAD_MARKER', arm=arm, marker=str(marker), error=repr(exc))
    return sorted(found)


def publish_index(api: HfApi, arm: str, config: dict, state: dict) -> None:
    entries = state.setdefault('checkpoints', {}).setdefault(arm, {})
    index = {
        'schema': 'chatterbox-s3gen-vc-grow-followup-checkpoint-index-v1',
        'repo': config['repo'], 'arm': arm, 'updated_utc': utc(),
        'reward': ('0.40*z(audiobox:CE)+0.40*z(-top2_emotion_MSE)'
                   '+0.20*z(target_CAMPPlus_cosine)' if arm == 'speaker_mix'
                   else '0.50*z(audiobox:CE)+0.50*z(-top2_emotion_MSE)'),
        'checkpoints': sorted(entries.values(), key=lambda row: row['step']),
    }
    api.upload_file(repo_id=config['repo'], repo_type='model',
                    path_or_fileobj=io.BytesIO((json.dumps(index, indent=2) + '\n').encode()),
                    path_in_repo='RL_CHECKPOINTS.json',
                    commit_message='Update verified RL checkpoint index')


def upload_checkpoints(api: HfApi, arm: str, config: dict, state: dict) -> bool:
    found = candidates(arm, config)
    saved = state.setdefault('checkpoints', {}).setdefault(arm, {})
    pending = [(step, file, path) for step, file, path in found
               if str(step) not in saved]
    if pending:
        remote = remote_files(api, config['repo'])
        for step, file, path in pending:
            digest = sha256(file)
            size = file.stat().st_size
            if path in remote and remote[path] != (digest, size):
                raise RuntimeError(f'Remote checkpoint conflict: {config["repo"]}/{path}')
            if path not in remote:
                emit('CHECKPOINT_UPLOAD_START', arm=arm, step=step, bytes=size)
                api.upload_file(repo_id=config['repo'], repo_type='model',
                                path_or_fileobj=str(file), path_in_repo=path,
                                commit_message=f'Publish complete RL checkpoint step {step}')
                remote = remote_files(api, config['repo'])
            if remote.get(path) != (digest, size):
                raise RuntimeError(f'Upload verification failed: {config["repo"]}/{path}')
            saved[str(step)] = {'arm': arm, 'step': step, 'sha256': digest,
                                'bytes': size, 'hf_path': path,
                                'verified_utc': utc()}
            state.setdefault('index_dirty', {})[arm] = True
            save_state(state)
            emit('CHECKPOINT_UPLOAD_VERIFIED', arm=arm, step=step,
                 sha256=digest, bytes=size)
    if pending or state.setdefault('index_dirty', {}).get(arm, False):
        publish_index(api, arm, config, state)
        state['index_dirty'][arm] = False
        save_state(state)
        emit('INDEX_UPDATED', arm=arm, count=len(saved))
    done_file = config['root'] / arm / 'DONE.json'
    if not done_file.exists():
        return False
    done = json.loads(done_file.read_text())
    return (done.get('status') == 'COMPLETE' and done.get('step') == 24000
            and bool(found) and all(str(step) in saved for step, _, _ in found))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--poll-seconds', type=int, default=30)
    args = parser.parse_args()
    if args.poll_seconds < 5:
        parser.error('--poll-seconds must be >= 5')
    state = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}
    if args.dry_run:
        for arm, config in RUNS.items():
            print(json.dumps({'arm': arm, 'repo': config['repo'],
                              'docs': sorted(public_files(arm, config)),
                              'checkpoints': [step for step, _, _ in candidates(arm, config)]}))
        return
    api = HfApi()
    while True:
        try:
            for arm, config in RUNS.items():
                bootstrap(api, arm, config, state)
            complete = [upload_checkpoints(api, arm, config, state)
                        for arm, config in RUNS.items()]
            if all(complete):
                emit('ALL_PUBLISHED')
                return
        except Exception as exc:
            emit('ERROR_RETRY', error=repr(exc))
            if args.once:
                raise
            time.sleep(60)
        if args.once:
            return
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
