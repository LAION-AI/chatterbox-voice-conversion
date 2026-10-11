#!/usr/bin/env python3
"""Submit the matched-seed evaluation after all training DONE markers.

One small login-node process; the actual inference runs on a four-GPU
Booster node. After its EVAL_DONE marker, publish a static HF Space with 40
packed audio containers. State and logs stay on Scratch, not in Home.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
OUT = Path('/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_followup_eval')
TRAIN_ROOTS = (
    Path('/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_v2_ce/continue'),
    Path('/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_v2_ce/remerged_new'),
    Path('/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_v3_ce_emo_speaker/speaker_mix'),
)
STATE = OUT / 'eval_watch_state.json'
LOG = OUT / 'eval_watch_log.jsonl'


def emit(event: str, **fields) -> None:
    row = {'utc': datetime.now(timezone.utc).isoformat(), 'event': event, **fields}
    line = json.dumps(row, sort_keys=True)
    print(line, flush=True)
    with LOG.open('a') as handle:
        handle.write(line + '\n')


def save(state: dict) -> None:
    temporary = STATE.with_suffix('.json.partial')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    os.replace(temporary, STATE)


def training_done() -> bool:
    for root in TRAIN_ROOTS:
        file = root / 'DONE.json'
        if not file.exists():
            return False
        marker = json.loads(file.read_text())
        if marker.get('status') != 'COMPLETE' or marker.get('step') != 24000:
            return False
    return True


def prepared() -> bool:
    file = OUT / 'prepared.COMPLETE.json'
    return file.exists() and json.loads(file.read_text()).get('status') == 'COMPLETE'


def smoke_ok() -> bool:
    file = OUT / 'EVAL_SMOKE.json'
    return file.exists() and json.loads(file.read_text()).get('status') == 'PASS'


def job_state(job_id: str) -> str:
    pending = subprocess.run(['/usr/bin/squeue', '-h', '-j', job_id, '-o', '%T'],
                             capture_output=True, text=True, check=True).stdout.strip()
    if pending:
        return pending.splitlines()[0]
    accounting = subprocess.run(['/usr/bin/sacct', '-j', job_id, '-n', '-P',
                                 '-o', 'State'], capture_output=True, text=True,
                                check=True).stdout.strip()
    return accounting.splitlines()[0].split('|')[0] if accounting else 'UNKNOWN'


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    state = json.loads(STATE.read_text()) if STATE.exists() else {'attempts': 0}
    last_report = None
    while True:
        try:
            if (OUT / 'SPACE_PUBLISHED.json').exists():
                emit('SPACE_PUBLISHED', url=json.loads(
                    (OUT / 'SPACE_PUBLISHED.json').read_text())['url'])
                return
            if (OUT / 'EVAL_DONE.json').exists():
                emit('SPACE_BUILD_OR_UPLOAD_START')
                with (OUT / 'space-publish.log').open('a') as log:
                    subprocess.run(['/bin/bash', str(HERE / 'publish_space.sh')],
                                   stdout=log, stderr=subprocess.STDOUT, check=True)
                emit('SPACE_BUILD_OR_UPLOAD_COMPLETE')
                continue
            ready = training_done() and prepared() and smoke_ok()
            if not ready:
                if last_report != 'WAITING_INPUTS':
                    emit('WAITING_INPUTS', training_done=training_done(),
                         prepared=prepared(), smoke_pass=smoke_ok())
                    last_report = 'WAITING_INPUTS'
                time.sleep(60)
                continue
            old_id = state.get('job_id')
            if old_id:
                status = job_state(old_id)
                if status in ('PENDING', 'CONFIGURING', 'RUNNING', 'COMPLETING'):
                    if last_report != status:
                        emit('EVAL_JOB_ACTIVE', job_id=old_id, status=status)
                        last_report = status
                    time.sleep(60)
                    continue
                if status == 'COMPLETED':
                    # Slurm accounting can precede visibility of the final
                    # filesystem marker. Allow one full poll before retry.
                    time.sleep(60)
                    if (OUT / 'EVAL_DONE.json').exists():
                        continue
                emit('EVAL_JOB_ENDED_WITHOUT_DONE', job_id=old_id, status=status)
                state.pop('job_id', None)
                save(state)
            if state['attempts'] >= 3:
                emit('EVAL_RETRY_LIMIT', attempts=state['attempts'])
                return
            result = subprocess.run(['/usr/bin/sbatch', '--parsable',
                                     str(HERE / 'eval_followups.sbatch')],
                                    capture_output=True, text=True, check=True)
            job_id = result.stdout.strip().split(';')[0]
            assert job_id.isdigit(), result.stdout
            state['job_id'] = job_id
            state['attempts'] += 1
            save(state)
            emit('EVAL_JOB_SUBMITTED', job_id=job_id, attempt=state['attempts'])
            last_report = None
            time.sleep(60)
        except Exception as exc:
            emit('ERROR_RETRY', error=repr(exc))
            time.sleep(120)


if __name__ == '__main__':
    main()
