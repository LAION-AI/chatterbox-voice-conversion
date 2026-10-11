#!/bin/bash
set -euo pipefail
source /e/home/jusers/schuhmann1/jupiter/chatterbox_s3gen_300h_v1/env.sh
# Booster nodes have no outbound HF access. This script is run by the small
# login-node watcher after the dependent GPU evaluation writes EVAL_DONE.json.
export HF_HOME=/e/home/jusers/schuhmann1/jupiter/.cache/huggingface
export HF_HUB_CACHE=/e/scratch/laionize/schuhmann1/hf_hub_cache
export HF_XET_CACHE=/e/scratch/laionize/schuhmann1/hf_xet_cache
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
exec python -u /e/home/jusers/schuhmann1/jupiter/chatterbox_s3gen_300h_v1/grow_v4/publish_after_full.py
