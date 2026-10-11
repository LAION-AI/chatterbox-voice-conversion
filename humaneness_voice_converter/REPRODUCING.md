# Reproducing Humaneness Voice Converter

This document records the actual research lineage, not an invented
one-command reproduction recipe. The public repository contains a complete
inference checkpoint and source snapshots; a full retrain additionally needs
the prepared, permission-compatible audio/token/score containers used here,
four GH200-class GPUs, and path configuration for the Slurm scripts. Never
expand a large corpus into millions of per-clip files: preparation and audio
evaluation use packed TAR/Parquet containers.

## Lineage and architecture

| Stage | Starting point | Trainable part | Purpose |
| --- | --- | --- | --- |
| Upstream | `ResembleAI/chatterbox`, revision `5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18` | none | S3 tokenizer, 192-d CAMPPlus, S3Gen flow and HiFT |
| Supervised 300.017 h | Upstream S3Gen | rank-128 q/k/v/out flow LoRA + 198→256→80 score projection | Learn 99 score axes from paired audio |
| Group-relative RL | Supervised adapter | flow adapter | Explore reward-guided acoustic variation |
| Speaker-Mix 24k | Completed earlier RL branch | rank-128 flow adapter | 40% CE, 40% emotion, 20% target speaker reward in groups of 16 |
| Full-flow 40k | Speaker-Mix 24k LoRA merged into its base | all 112,620,864 flow parameters | 40% CE, 30% emotion, 30% target speaker reward; the release checkpoint |

The supervised selection contains 111,665 unique UIDs and 300.017 hours,
including about 30.003 hours of synthetic sidecar data. It was drawn from
the S6/tier-2 selection, not from the Chatterbox text model. See
`training/base/choose_s6.py`, `prepare.py` and `train.py` for selection,
tokenization, speaker, mel and score preparation. The public base training
code and prior optimizer checkpoints are also archived at
[`laion/chatterbox-s3gen-vc-grow`](https://huggingface.co/laion/chatterbox-s3gen-vc-grow).
Input selection for the RL stages used strong examples from 40 emotion axes;
it mixed recorded and synthetic audio. It is **not** an all-real or
speaker-disjoint training set. The 99-axis ordering is fixed in
`conditioning/score_order.json` and must not be permuted.

The full-flow stage loads the verified 24k Speaker-Mix checkpoint from the
preceding RL stage, folds 224 attention LoRA updates into the flow weights,
and verifies a maximum merge-output discrepancy of `2.861e-6`. The frozen
S3 tokenizer, CAMPPlus speaker encoder, HiFT vocoder and Ears judge do not
receive gradients. The full flow generator, including its score projection,
does. The exact code is `training/grow_v4/train_full.py`; the source paths
default to Jupiter Scratch and require adaptation if the prepared containers
are stored elsewhere.

## Full-flow objective and schedule

Each training condition has a source speech recording and target voice. The
model draws **16** stochastic flow candidates per group across four GPUs.
Humaneness Ears Medium (revision
`818506970d93c809a3295a0a02c40dee8ff2bfc3` of
[`laion/humaneness-ears-base-medium`](https://huggingface.co/laion/humaneness-ears-base-medium))
predicts AudioBox Content Enjoyment and 40 emotion axes. The two source
emotion axes with the highest predicted scores are fixed for that condition;
candidate emotion error is their mean squared error against the source.
CAMPPlus cosine compares each output to the target voice embedding.

Within **each 16-candidate group**, the three reward components are
individually standardized, then mixed as

```text
0.40 × z(AudioBox Content Enjoyment)
− 0.30 × z(source-top-two emotion MSE)
+ 0.30 × z(target CAMPPlus cosine)
```

The signed group-relative advantages weight the continuous-flow matching
objective. A frozen starting-policy velocity anchor has coefficient `0.025`.
This is GROW-inspired group-relative reinforcement learning for a flow
decoder, **not** token-level GRPO. See the implementation in
`training/grow_v4/train_full.py` and the preceding `grow_v1`/`grow_v2`/`grow_v3`
modules; follow the code for details such as gradient accumulation and
conditioning masks.

The full-flow schedule spans **40,000 groups**: 2,000-group linear warm-up,
peak LR `3e-7`, then cosine decay to `3e-8`. It does **not** restart when
Slurm's time limit splits the run into five 8,000-group segments. AdamW
optimizer and RNG states are saved every 1,000 groups, so those segment
boundaries can resume without resetting training. The run used the JUPITER
`booster` partition, one node with four GH200 GPUs, and 10h30m per segment.
The Slurm/monitor sources are in `training/grow_v4/`. The final completed
optimizer checkpoint is `step-0040000.pt`; the public `model.safetensors` was
exported by `training/export_full.py` from this checkpoint after checking
its complete-proof metadata. The SafeTensors inference file has no optimizer
state. The Jupiter job IDs were 2260781 through 2260785; final completion
was verified on 11 October 2026.

## What is actually in the release

`model.safetensors` is the complete S3Gen state (not just adapter weights).
`conditioning/ears-medium/` includes the frozen Ears Medium score model,
Whisper configuration, feature-extractor configuration, classes and score
normalization. The S3Gen weights derive from upstream Chatterbox revision
above; its original `s3gen.safetensors` had SHA-256
`2b78103c654207393955e4900aac14a12de8ef25f4b09424f1ef91941f161d4e`.
`EXPORT_PROOF.json` records the SHA-256 and tensor count of the exported
merged file. The model card's CC BY 4.0 designation applies to LAION's
contributions; the upstream MIT notice is retained in `NOTICE.md`.

The tested `infer.py` takes raw audio. It extracts S3 tokens from source
audio, predicts source/target scores with Ears Medium, puts stable target
identity/timbre dimensions into the source score vector, computes the target
CAMPPlus vector and acoustic prefix, runs the score-conditioned flow for 10
steps, and synthesizes 24-kHz audio through HiFT. Source clips of 1–30 s are
accepted by this CLI; 5–10-s single-speaker targets are recommended. The
target prefix is capped at 250 S3 tokens, or about 10 s. Input normalization,
resampling and the exact stable target-score dimensions are visible in
`infer.py`. A GPU inference smoke test completed successfully with a 6.528-s
source, 5.448-s target and 6.52-s output.

## Evaluation and limitations

The matched evaluation code is `training/grow_v4/eval_after_full.py` and
`finalize_eval.py`. The first produces 160 source cases × two target voices
with matched seeds and packed MP3 TARs; the second verifies all case/target
keys, checkpoint completion and metric rows. It reports 93 *other*,
training-UID-disjoint source cases, but speaker-level disjointness was not
verified. The curated listening demo chooses 40 source cases **by source
recording quality**, with English preferred and German as fallback, then
reuses exactly those 40 and the same A/B targets across models. Ten curated
source UIDs are known to overlap the RL source selection. These are
transparent listening examples, not an independent benchmark.

Humaneness Ears Medium supplies CE/PQ/emotion proxies; it is also part of
the RL training reward. Thus an increase in those numbers can partly reflect
reward-model optimization. CAMPPlus cosine is a speaker-embedding proxy,
not proof of perceptual identity. Vanilla Chatterbox scores *higher* target
cosine than the 40k model on the curated set. There is no blinded human MOS,
formal cross-language benchmark or WER claim for this release. Any external
converter numbers in the demo must be generated from the same 40 source clips
and two targets, not copied from different papers or datasets.

The best-of-N study uses the 40k model with 16 distinct, deterministic seeds
per source/target pair. Pools of size 1, 4, 8 and 16 are nested. It ranks
candidates by **equal thirds** of standardized CE, target cosine and negative
top-two emotion MSE. Those standardization constants are fixed from the
separate 93 training-UID-disjoint sources; they are not refitted to each
candidate group or pool size. The release notes preserve every candidate's
metrics and seed, while audio is kept in packed TARs. Since selection uses
the same judge family as training, increasing N is guaranteed only to
increase the *selected proxy reward*, not human quality. The measured
generation/scoring times are sequential GPU timings; batching can be
implemented and may change throughput, but a speedup is not claimed without
measurement.

## Reproduction checklist

1. Install the Python/CUDA dependencies in `requirements.txt` and test
   `infer.py` with one authorized source/target pair.
2. For a full retrain, acquire an appropriately licensed S6/tier-2-style
   source selection and the prepared packed semantic-token, mel, score and
   reference fields. Verify 99-axis score order and training/holdout UID
   boundaries. See `training/base/` for selection and preparation code.
3. Run the supervised 300h pilot, then the intermediate RL stages in the
   published lineage. Keep optimizer-bearing checkpoints and audit manifests.
4. Merge the verified 24k Speaker-Mix adapter, check numerical equivalence,
   then run `training/grow_v4/run.sbatch` in five resumable 8k segments,
   preserving one global 40k warm-up/cosine schedule.
5. Verify the final `.COMPLETE.json` and `DONE.json`, export full
   `model.safetensors`, run the raw-audio smoke test and the same-source
   evaluation. Do not report a Slurm submission as a completed model.

These scripts include original machine-specific paths, and some upstream
models must be fetched separately under their own terms. They are supplied
to make the method auditable and adaptable, not to imply that data licenses,
network access or a four-GPU cluster are bundled in this repository.
