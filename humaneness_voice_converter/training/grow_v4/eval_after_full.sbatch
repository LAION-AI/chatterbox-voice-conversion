#!/bin/bash
#SBATCH --job-name=cbx-grow-v4-eval
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --time=03:00:00
#SBATCH --output=/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_v4_full_403030/evaluation/eval-%j.out
#SBATCH --error=/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_v4_full_403030/evaluation/eval-%j.err
set -euo pipefail
source /e/home/jusers/schuhmann1/jupiter/chatterbox_s3gen_300h_v1/env.sh
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
args=()
if [[ ${CHATTERBOX_GROW_V4_EVAL_SMOKE:-0} = 1 ]]; then
  args+=(--smoke --smoke-step 35000)
fi
srun --ntasks=4 --gpus-per-task=1 --kill-on-bad-exit=1 --cpu-bind=none \
  python -u /e/home/jusers/schuhmann1/jupiter/chatterbox_s3gen_300h_v1/grow_v4/eval_after_full.py "${args[@]}"
