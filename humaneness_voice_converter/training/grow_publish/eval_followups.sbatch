#!/bin/bash
#SBATCH --job-name=cbx-grow-eval
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --time=11:50:00
#SBATCH --output=/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_followup_eval/eval-%j.out
#SBATCH --error=/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_followup_eval/eval-%j.err
set -euo pipefail
source /e/home/jusers/schuhmann1/jupiter/chatterbox_s3gen_300h_v1/env.sh
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
srun --ntasks=4 --gpus-per-task=1 --kill-on-bad-exit=1 --cpu-bind=none \
  python -u /e/home/jusers/schuhmann1/jupiter/chatterbox_s3gen_300h_v1/grow_publish/eval_followups.py
