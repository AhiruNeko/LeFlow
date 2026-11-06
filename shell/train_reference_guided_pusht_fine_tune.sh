#!/usr/bin/env bash
set -euo pipefail

export STABLEWM_HOME="${STABLEWM_HOME:-/root/autodl-tmp/pusht}"

DATA=pusht
LEWM_CHECKPOINT="${LEWM_CHECKPOINT:-$STABLEWM_HOME/lewm_object.ckpt}"
# Produced by the staged distillation run; override for the desired seed/run.
DISTILLATION_CHECKPOINT="${DISTILLATION_CHECKPOINT:-$STABLEWM_HOME/reference_guided_world_model/pusht_h10/reference_guided_world_model_distillation_best.pt}"
ROLLOUT_CACHE_PATH="${ROLLOUT_CACHE_PATH:-$STABLEWM_HOME/reference_guided_world_model/pusht_h10_staged_rollout_cache.h5}"
SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-reference_guided_world_model/pusht_h10_fine_tune}"

conda run -n lewm --no-capture-output python train_reference_guided_world_model.py \
  "data=$DATA" \
  "lewm_checkpoint=$LEWM_CHECKPOINT" \
  "rollout_cache.path=$ROLLOUT_CACHE_PATH" \
  "seed=$SEED" \
  "subdir=$OUTPUT_DIR" \
  "staged_training.generate_rollout_cache=false" \
  "staged_training.distillation=false" \
  "staged_training.fine_tune=true" \
  "staged_training.distillation_checkpoint=$DISTILLATION_CHECKPOINT"
