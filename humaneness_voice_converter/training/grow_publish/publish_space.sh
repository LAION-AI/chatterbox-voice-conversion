#!/bin/bash
set -euo pipefail
export PYTHONPATH=${PYTHONPATH:-}
source /e/home/jusers/schuhmann1/jupiter/chatterbox_s3gen_300h_v1/env.sh
# Training uses the Scratch model cache offline; publishing needs the user's
# authenticated HF token in the Home cache and online API access.
export HF_HOME=/e/home/jusers/schuhmann1/jupiter/.cache/huggingface
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
exec python -u /e/home/jusers/schuhmann1/jupiter/chatterbox_s3gen_300h_v1/grow_publish/build_and_publish_space.py
