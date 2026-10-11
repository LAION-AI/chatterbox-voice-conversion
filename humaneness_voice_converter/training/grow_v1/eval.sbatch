#!/bin/bash
#SBATCH --job-name=cbx-grow-compare
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_v1/logs/eval_%j.out
#SBATCH --error=/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_v1/logs/eval_%j.err
set -euo pipefail
source /e/home/jusers/schuhmann1/jupiter/chatterbox_s3gen_300h_v1/env.sh
cd /e/scratch/reformo/schuhmann1_moss/out/chatterbox_s3gen_300h_v1/grow_v1
TRAIN_ROOT=/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_v1/training
for arm in continue merged_new; do
  test -f "$TRAIN_ROOT/$arm/DONE.json"
done
srun --nodes=1 --ntasks=1 --gpus-per-task=1 --cpu-bind=none \
  python -u /e/home/jusers/schuhmann1/jupiter/chatterbox_s3gen_300h_v1/grow_v1/eval_compare.py
