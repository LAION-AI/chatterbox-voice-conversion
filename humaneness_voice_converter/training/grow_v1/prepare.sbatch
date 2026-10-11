#!/bin/bash
#SBATCH --job-name=cbx-grow-pool-ears
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=/e/scratch/reformo/schuhmann1_moss/out/chatterbox_s3gen_300h_v1/grow_v1/logs/prepare_%j.out
#SBATCH --error=/e/scratch/reformo/schuhmann1_moss/out/chatterbox_s3gen_300h_v1/grow_v1/logs/prepare_%j.err
set -euo pipefail
source /e/home/jusers/schuhmann1/jupiter/chatterbox_s3gen_300h_v1/env.sh
HERE=/e/home/jusers/schuhmann1/jupiter/chatterbox_s3gen_300h_v1/grow_v1
cd /e/scratch/reformo/schuhmann1_moss/out/chatterbox_s3gen_300h_v1/grow_v1
python -u "$HERE/prepare_pool.py"
srun --nodes=1 --ntasks=4 --ntasks-per-node=4 --gpus-per-task=1 --kill-on-bad-exit=1 --cpu-bind=none \
  python -u "$HERE/score_inputs.py"
