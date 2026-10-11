#!/bin/bash
# Submit five sequential 8k-group jobs only after the 16-group smoke passes.
set -euo pipefail
cd /e/home/jusers/schuhmann1/jupiter
root=/e/scratch/laionize/schuhmann1/chatterbox_s3gen_grow_v4_full_403030
test "$(jq -r .status "$root/SMOKE.json")" = PASS
test "$(jq -r .groups_tested "$root/SMOKE.json")" = 16
previous=
for segment in 1 2 3 4 5; do
  if [[ -n "$previous" ]]; then
    job=$(sbatch --parsable --dependency="afterok:$previous" \
      --export=ALL,CHATTERBOX_GROW_V4_MODE=full \
      chatterbox_s3gen_300h_v1/grow_v4/run.sbatch)
  else
    job=$(sbatch --parsable --export=ALL,CHATTERBOX_GROW_V4_MODE=full \
      chatterbox_s3gen_300h_v1/grow_v4/run.sbatch)
  fi
  printf 'segment=%d job=%s dependency=%s\n' "$segment" "$job" "${previous:-none}"
  previous=$job
done
