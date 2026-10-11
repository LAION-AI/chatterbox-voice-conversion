#!/bin/bash
set -euo pipefail
source /e/home/jusers/schuhmann1/jupiter/chatterbox_s3gen_300h_v1/env.sh
# The training environment is intentionally offline; the login-node publisher
# must use the user's existing Hugging Face login and network access instead.
unset HF_HOME HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export HF_HUB_DISABLE_PROGRESS_BARS=1
exec python -u /e/home/jusers/schuhmann1/jupiter/chatterbox_s3gen_300h_v1/grow_v1/upload_checkpoints.py "$@"
