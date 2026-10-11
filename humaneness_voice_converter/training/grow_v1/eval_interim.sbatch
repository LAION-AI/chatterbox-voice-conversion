#!/bin/bash
#SBATCH --job-name=cbx-grow-interim-listen
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_v1/logs/interim_%j.out
#SBATCH --error=/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_v1/logs/interim_%j.err
set -euo pipefail
ulimit -c 0
source /e/home/jusers/schuhmann1/jupiter/chatterbox_s3gen_300h_v1/env.sh
cd /e/scratch/reformo/schuhmann1_moss/out/chatterbox_s3gen_300h_v1/grow_v1
srun --nodes=1 --ntasks=1 --gpus-per-task=1 --cpu-bind=none \
  python -u /e/home/jusers/schuhmann1/jupiter/chatterbox_s3gen_300h_v1/grow_v1/eval_compare.py --interim
