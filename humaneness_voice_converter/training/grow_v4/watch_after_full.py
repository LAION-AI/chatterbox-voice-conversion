#!/usr/bin/env python3
"""Login-node watcher: retry the dependent eval once, then publish the Space.

The four-GPU evaluation is submitted with Slurm afterok on the final training
segment. Compute nodes cannot reach Hugging Face, so only this lightweight
login-node process performs the final packed-audio and HTML upload.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
OUT = Path('/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_v4_full_403030')
EVAL = OUT / 'evaluation'
STATE = EVAL / 'AUTO_EVAL_STATE.json'
EVENTS = EVAL / 'AUTO_EVAL_EVENTS.jsonl'
TRAIN_JOB = '2260785'


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    os.replace(temporary, path)


def emit(event: str, **fields) -> None:
    row = {'utc': datetime.now(timezone.utc).isoformat(), 'event': event, **fields}
    with EVENTS.open('a') as handle:
        handle.write(json.dumps(row, sort_keys=True) + '\n')
    print(json.dumps(row), flush=True)


def job_state(job_id: str) -> str:
    queue = subprocess.run(['/usr/bin/squeue', '-h', '-j', job_id, '-o', '%T'],
                           capture_output=True, text=True, check=True)
    if queue.stdout.strip():
        return queue.stdout.strip().splitlines()[0].split()[0]
    accounting = subprocess.run(['/usr/bin/sacct', '-j', job_id, '-n', '-P',
                                 '-o', 'State'], capture_output=True,
                                text=True, check=True)
    for line in accounting.stdout.splitlines():
        if line.strip():
            return line.split('|')[0].split()[0].split('+')[0]
    return 'UNKNOWN'


def training_done() -> bool:
    marker = OUT / 'DONE.json'
    return marker.exists() and read_json(marker).get('status') == 'COMPLETE' \
        and read_json(marker).get('step') == 40000


def eval_done() -> bool:
    marker = EVAL / 'EVAL_DONE.json'
    return marker.exists() and read_json(marker).get('status') == 'COMPLETE' \
        and read_json(marker).get('step') == 40000


def main() -> None:
    EVAL.mkdir(parents=True, exist_ok=True)
    assert STATE.exists(), 'Submit the Slurm evaluation before the watcher'
    state = read_json(STATE)
    last = None
    while True:
        if (EVAL / 'SPACE_PUBLISHED.json').exists():
            proof = read_json(EVAL / 'SPACE_PUBLISHED.json')
            assert proof['status'] == 'COMPLETE'
            state['status'] = 'COMPLETE'
            state['space_url'] = proof['url']
            state['revision'] = proof['revision']
            atomic_json(STATE, state)
            emit('COMPLETE', url=proof['url'], revision=proof['revision'])
            return
        if eval_done():
            state['status'] = 'PUBLISHING'
            atomic_json(STATE, state)
            try:
                with (EVAL / 'space-publish.log').open('a') as handle:
                    subprocess.run(['/bin/bash', str(HERE / 'publish_after_full.sh')],
                                   stdout=handle, stderr=subprocess.STDOUT,
                                   check=True, timeout=1200)
                emit('PUBLISH_COMMAND_COMPLETE')
                continue
            except Exception as error:
                state['publish_attempts'] = int(state.get('publish_attempts', 0)) + 1
                atomic_json(STATE, state)
                emit('PUBLISH_RETRY', attempt=state['publish_attempts'],
                     error=repr(error))
                if state['publish_attempts'] >= 3:
                    state['status'] = 'PUBLISH_FAILED'
                    atomic_json(STATE, state)
                    return
                time.sleep(60)
                continue
        train_status = job_state(TRAIN_JOB)
        if train_status in ('FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL',
                            'OUT_OF_MEMORY'):
            state['status'] = 'TRAIN_FAILED'
            state['train_state'] = train_status
            atomic_json(STATE, state)
            emit('TRAIN_FAILED', slurm_state=train_status)
            return
        job_id = str(state['eval_job_id'])
        status = job_state(job_id)
        if status == 'UNKNOWN':
            time.sleep(60)
            continue
        if status in ('PENDING', 'CONFIGURING', 'RUNNING', 'COMPLETING'):
            if status != last:
                emit('EVAL_ACTIVE', job_id=job_id, slurm_state=status,
                     training_done=training_done())
                last = status
            time.sleep(30)
            continue
        if status == 'COMPLETED':
            # Slurm accounting may lead filesystem visibility by a few seconds.
            state['completed_without_marker_polls'] = int(
                state.get('completed_without_marker_polls', 0)) + 1
            atomic_json(STATE, state)
            if state['completed_without_marker_polls'] <= 4:
                time.sleep(30)
                continue
        state.pop('completed_without_marker_polls', None)
        state['eval_attempts'] = int(state.get('eval_attempts', 1)) + 1
        if not training_done() or state['eval_attempts'] > 2:
            state['status'] = 'EVAL_FAILED'
            state['eval_slurm_state'] = status
            atomic_json(STATE, state)
            emit('EVAL_FAILED', job_id=job_id, slurm_state=status)
            return
        result = subprocess.run(['/usr/bin/sbatch', '--parsable',
                                 str(HERE / 'eval_after_full.sbatch')],
                                capture_output=True, text=True, check=True)
        replacement = result.stdout.strip().split(';')[0]
        assert replacement.isdigit(), result.stdout
        state['eval_job_id'] = replacement
        state['status'] = 'EVAL_RETRY'
        atomic_json(STATE, state)
        emit('EVAL_RETRY_SUBMITTED', old_job_id=job_id,
             new_job_id=replacement, old_state=status)
        last = None
        time.sleep(30)


if __name__ == '__main__':
    main()
