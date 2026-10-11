#!/bin/bash
#SBATCH --job-name=cbx-grow-eval-prep
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_followup_eval/prep-%j.out
#SBATCH --error=/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_followup_eval/prep-%j.err
set -euo pipefail
source /e/home/jusers/schuhmann1/jupiter/chatterbox_s3gen_300h_v1/env.sh
python -u /e/home/jusers/schuhmann1/jupiter/chatterbox_s3gen_300h_v1/grow_publish/prepare_site_inputs.py
