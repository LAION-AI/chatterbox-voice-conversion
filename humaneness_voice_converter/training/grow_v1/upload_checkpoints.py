#!/usr/bin/env python3
"""Publish each completed two-arm Chatterbox GROW checkpoint from the login node.

Only a `step-*.COMPLETE.json` marker followed by a size-matching `.pt` is
eligible. Hugging Face LFS SHA-256 is checked before a file is recorded as
published. The loop exits after both step-4000 DONE markers and every marked
checkpoint have been verified online. No model inference or audio decoding.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

from huggingface_hub import HfApi

ROOT = Path(os.environ.get('CHATTERBOX_GROW_TRAIN_ROOT',
    '/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_v1/training'))
HERE = Path(__file__).resolve().parent
STATE = HERE / 'HF_CHECKPOINT_UPLOAD_STATE.json'
LOG = HERE / 'HF_CHECKPOINT_UPLOAD_LOG.jsonl'
REPO = 'laion/chatterbox-s3gen-vc-grow'
ARMS = ('continue', 'merged_new')


def emit(event: str, **fields) -> None:
    row = {'utc': datetime.now(timezone.utc).isoformat(), 'event': event, **fields}
    line = json.dumps(row, sort_keys=True)
    print(line, flush=True)
    with LOG.open('a') as stream:
        stream.write(line + '\n')


def atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def candidates() -> list[tuple[str, int, Path, str]]:
    found = []
    for arm in ARMS:
        directory = ROOT / arm / 'checkpoints'
        for marker in directory.glob('step-*.COMPLETE.json'):
            try:
                proof = json.loads(marker.read_text())
                step = int(marker.name.split('.')[0].split('-')[1])
                file = directory / f'step-{step:07d}.pt'
                if (proof.get('status') != 'COMPLETE' or proof.get('arm') != arm
                        or proof.get('step') != step or not file.is_file()
                        or file.stat().st_size != proof.get('bytes')):
                    emit('SKIP_INCOMPLETE', arm=arm, step=step)
                    continue
                found.append((arm, step, file,
                              f'checkpoints/rl/{arm}/step-{step:07d}.pt'))
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                emit('SKIP_BAD_MARKER', marker=str(marker), error=str(exc))
    return sorted(found, key=lambda x: (x[1], x[0]))


def remote_files(api: HfApi) -> dict[str, tuple[str | None, int | None]]:
    info = api.model_info(REPO, files_metadata=True)
    return {f.rfilename: (f.lfs.sha256 if f.lfs else None, f.size)
            for f in info.siblings}


def publish_index(api: HfApi, state: dict) -> None:
    # The public index contains no local paths, hostnames, or credentials.
    index = {'schema': 'chatterbox-s3gen-vc-grow-checkpoint-index-v1',
             'repo': REPO, 'updated_utc': datetime.now(timezone.utc).isoformat(),
             'checkpoints': sorted(state['checkpoints'].values(),
                                   key=lambda row: (row['arm'], row['step']))}
    index_path = HERE / 'HF_CHECKPOINT_INDEX_PUBLIC.json'
    atomic_json(index_path, index)
    api.upload_file(repo_id=REPO, repo_type='model', path_or_fileobj=str(index_path),
                    path_in_repo='RL_CHECKPOINTS.json',
                    commit_message='Update verified RL checkpoint index')


def all_done(state: dict, found: list[tuple[str, int, Path, str]]) -> bool:
    if not found:
        return False
    for arm in ARMS:
        done_file = ROOT / arm / 'DONE.json'
        if not done_file.is_file():
            return False
        done = json.loads(done_file.read_text())
        if done.get('status') != 'COMPLETE' or done.get('step') != 4000:
            return False
    return all(f'{arm}/{step:07d}' in state['checkpoints'] for arm, step, _, _ in found)


def pass_once(api: HfApi, state: dict, dry_run: bool) -> bool:
    found = candidates()
    if dry_run:
        for arm, step, file, hf_path in found:
            print(json.dumps({'arm': arm, 'step': step, 'bytes': file.stat().st_size,
                              'hf_path': hf_path, 'already_recorded':
                              f'{arm}/{step:07d}' in state['checkpoints']}))
        return False
    pending = any(f'{arm}/{step:07d}' not in state['checkpoints']
                  for arm, step, _, _ in found)
    if not pending and not state.get('index_dirty', False):
        return all_done(state, found)
    remote = remote_files(api)
    for arm, step, file, hf_path in found:
        key = f'{arm}/{step:07d}'
        if key in state['checkpoints']:
            expected = state['checkpoints'][key]['sha256']
            if remote.get(hf_path) == (expected, file.stat().st_size):
                continue
            emit('REMOTE_MISSING_OR_MISMATCH', arm=arm, step=step, hf_path=hf_path)
            # Do not silently overwrite or forget previously published data.
            raise RuntimeError(f'Remote state changed unexpectedly: {hf_path}')
        digest = sha256(file)
        if hf_path in remote:
            if remote[hf_path] != (digest, file.stat().st_size):
                raise RuntimeError(f'Remote checkpoint differs: {hf_path}')
            emit('FOUND_ALREADY_REMOTE', arm=arm, step=step, sha256=digest)
        else:
            emit('UPLOAD_START', arm=arm, step=step, bytes=file.stat().st_size,
                 hf_path=hf_path)
            api.upload_file(repo_id=REPO, repo_type='model', path_or_fileobj=str(file),
                            path_in_repo=hf_path,
                            commit_message=f'Publish complete {arm} RL checkpoint step {step}')
            remote = remote_files(api)
            if remote.get(hf_path) != (digest, file.stat().st_size):
                raise RuntimeError(f'Upload verification failed: {hf_path}')
            emit('UPLOAD_VERIFIED', arm=arm, step=step, sha256=digest,
                 hf_path=hf_path)
        state['checkpoints'][key] = {'arm': arm, 'step': step, 'sha256': digest,
                                     'bytes': file.stat().st_size,
                                     'hf_path': hf_path,
                                     'verified_utc': datetime.now(timezone.utc).isoformat()}
        state['index_dirty'] = True
        atomic_json(STATE, state)
    if state.get('index_dirty', False):
        publish_index(api, state)
        state['index_dirty'] = False
        atomic_json(STATE, state)
        emit('INDEX_UPDATED', checkpoint_count=len(state['checkpoints']))
    return all_done(state, found)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--poll-seconds', type=int, default=20)
    args = parser.parse_args()
    assert args.poll_seconds >= 5
    state = json.loads(STATE.read_text()) if STATE.exists() else {'checkpoints': {},
                                                                  'index_dirty': False}
    if args.dry_run:
        pass_once(HfApi(), state, True)
        return
    api = HfApi()
    while True:
        try:
            if pass_once(api, state, False):
                emit('ALL_CHECKPOINTS_PUBLISHED', count=len(state['checkpoints']))
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
