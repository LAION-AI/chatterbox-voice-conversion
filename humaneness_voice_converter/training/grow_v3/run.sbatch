#!/bin/bash
#SBATCH --job-name=cbx-grow-v3-spk
#SBATCH --account=reformo
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=12
#SBATCH --time=11:50:00
#SBATCH --output=/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_v3_ce_emo_speaker/logs/%x-%j.out
#SBATCH --error=/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_v3_ce_emo_speaker/logs/%x-%j.err
set -euo pipefail
source /e/home/jusers/schuhmann1/jupiter/chatterbox_s3gen_300h_v1/env.sh
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
mode=${CHATTERBOX_GROW_V3_MODE:?Set smoke or full}
test "$mode" = smoke || test "$mode" = full
export CHATTERBOX_GROW_V3_ROOT=${CHATTERBOX_GROW_V3_ROOT:-/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_v3_ce_emo_speaker}
if [[ "$mode" = full && -f "$CHATTERBOX_GROW_V3_ROOT/speaker_mix/DONE.json" ]]; then
  if [[ $(jq -r .step "$CHATTERBOX_GROW_V3_ROOT/speaker_mix/DONE.json") = 24000 ]]; then
    echo "Already complete: speaker_mix step 24000"
    exit 0
  fi
fi
args=(--mode "$mode")
if [[ "$mode" = full && -f "$CHATTERBOX_GROW_V3_ROOT/speaker_mix/checkpoints/LATEST.json" ]]; then
  resume=$(jq -r .file "$CHATTERBOX_GROW_V3_ROOT/speaker_mix/checkpoints/LATEST.json")
  test -f "$resume"
  args+=(--resume "$resume")
fi
srun --ntasks=4 --gpus-per-task=1 --kill-on-bad-exit=1 --cpu-bind=none \
  python -u /e/home/jusers/schuhmann1/jupiter/chatterbox_s3gen_300h_v1/grow_v3/train_speaker.py "${args[@]}"
