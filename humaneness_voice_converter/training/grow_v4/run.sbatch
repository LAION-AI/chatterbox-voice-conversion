#!/bin/bash
#SBATCH --job-name=cbx-grow-v4-full
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --time=10:30:00
#SBATCH --output=/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_v4_full_403030/logs/%x-%j.out
#SBATCH --error=/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_v4_full_403030/logs/%x-%j.err
set -euo pipefail
source /e/home/jusers/schuhmann1/jupiter/chatterbox_s3gen_300h_v1/env.sh
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
mode=${CHATTERBOX_GROW_V4_MODE:?Set smoke or full}
test "$mode" = smoke || test "$mode" = full
export CHATTERBOX_GROW_V4_ROOT=${CHATTERBOX_GROW_V4_ROOT:-/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_v4_full_403030}
if [[ "$mode" = full && -f "$CHATTERBOX_GROW_V4_ROOT/DONE.json" ]]; then
  echo 'Already complete: full-flow 40000 groups'
  exit 0
fi
args=(--mode "$mode")
if [[ "$mode" = full && -f "$CHATTERBOX_GROW_V4_ROOT/checkpoints/LATEST.json" ]]; then
  resume=$(jq -r .file "$CHATTERBOX_GROW_V4_ROOT/checkpoints/LATEST.json")
  test -f "$resume"
  args+=(--resume "$resume")
fi
srun --ntasks=4 --gpus-per-task=1 --kill-on-bad-exit=1 --cpu-bind=none \
  python -u /e/home/jusers/schuhmann1/jupiter/chatterbox_s3gen_300h_v1/grow_v4/train_full.py "${args[@]}"
