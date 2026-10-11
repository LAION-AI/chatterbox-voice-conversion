#!/bin/bash
#SBATCH --job-name=cbx-grow-2arm-smoke
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=/e/scratch/reformo/schuhmann1_moss/out/chatterbox_s3gen_300h_v1/grow_v1/logs/smoke_%j.out
#SBATCH --error=/e/scratch/reformo/schuhmann1_moss/out/chatterbox_s3gen_300h_v1/grow_v1/logs/smoke_%j.err
set -euo pipefail
source /e/home/jusers/schuhmann1/jupiter/chatterbox_s3gen_300h_v1/env.sh
cd /e/scratch/reformo/schuhmann1_moss/out/chatterbox_s3gen_300h_v1/grow_v1
srun --nodes=1 --ntasks=2 --ntasks-per-node=2 --gpus-per-task=1 --kill-on-bad-exit=1 --cpu-bind=none \
  bash /e/home/jusers/schuhmann1/jupiter/chatterbox_s3gen_300h_v1/grow_v1/run_arm.sh smoke
