#!/bin/bash
#SBATCH --job-name=cbx-grow-v2
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --time=11:50:00
#SBATCH --output=/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_v2_ce/logs/%x-%j.out
#SBATCH --error=/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_v2_ce/logs/%x-%j.err
set -euo pipefail
source /e/home/jusers/schuhmann1/jupiter/chatterbox_s3gen_300h_v1/env.sh
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
arm=${CHATTERBOX_GROW_V2_ARM:?Set continue or remerged_new}
mode=${CHATTERBOX_GROW_V2_MODE:?Set smoke or full}
test "$arm" = continue || test "$arm" = remerged_new
test "$mode" = smoke || test "$mode" = full
export CHATTERBOX_GROW_V2_ROOT=${CHATTERBOX_GROW_V2_ROOT:-/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_v2_ce}
export CHATTERBOX_GROW_V2_REWARD_SCORE=${CHATTERBOX_GROW_V2_REWARD_SCORE:-audiobox:CE}
if [[ "$mode" = full && -f "$CHATTERBOX_GROW_V2_ROOT/$arm/DONE.json" ]]; then
  if [[ $(jq -r .step "$CHATTERBOX_GROW_V2_ROOT/$arm/DONE.json") = 24000 ]]; then
    echo "Already complete: $arm step 24000"
    exit 0
  fi
fi
args=(--arm "$arm" --mode "$mode")
if [[ "$mode" = full && -f "$CHATTERBOX_GROW_V2_ROOT/$arm/checkpoints/LATEST.json" ]]; then
  resume=$(jq -r .file "$CHATTERBOX_GROW_V2_ROOT/$arm/checkpoints/LATEST.json")
  test -f "$resume"
  args+=(--resume "$resume")
fi
srun --ntasks=4 --gpus-per-task=1 --kill-on-bad-exit=1 --cpu-bind=none \
  python -u /e/home/jusers/schuhmann1/jupiter/chatterbox_s3gen_300h_v1/grow_v2/train_ddp.py "${args[@]}"
