#!/bin/bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
TRAIN_ROOT=${CHATTERBOX_GROW_TRAIN_ROOT:-/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_v1/training}
if [[ "${SLURM_NTASKS:-}" == 1 ]]; then
  arm=${CHATTERBOX_GROW_ARM:?Set continue or merged_new for one-task jobs}
elif [[ "$SLURM_PROCID" == 0 ]]; then
  arm=continue
elif [[ "$SLURM_PROCID" == 1 ]]; then
  arm=merged_new
else
  echo "Expected two workers, got rank $SLURM_PROCID" >&2
  exit 2
fi
mode="${1:?smoke or full required}"
args=(--mode "$mode" --arm "$arm")
if [[ "$mode" == full && -n "${CHATTERBOX_GROW_RESUME:-}" ]]; then
  test -f "$CHATTERBOX_GROW_RESUME"
  args+=(--resume "$CHATTERBOX_GROW_RESUME")
elif [[ "$mode" == full && -f "$TRAIN_ROOT/$arm/checkpoints/LATEST.json" ]]; then
  step=$(jq -r .step "$TRAIN_ROOT/$arm/checkpoints/LATEST.json")
  printf -v resume '%s/%s/checkpoints/step-%07d.pt' "$TRAIN_ROOT" "$arm" "$step"
  test -f "$resume"
  args+=(--resume "$resume")
fi
exec python -u "$HERE/train_grow.py" "${args[@]}"
