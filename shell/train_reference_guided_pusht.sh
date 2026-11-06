#!/usr/bin/env bash
set -euo pipefail

export STABLEWM_HOME="${STABLEWM_HOME:-/root/autodl-tmp/pusht}"

DATA=pusht
PLANNER_CHECKPOINT="${PLANNER_CHECKPOINT:-$STABLEWM_HOME/leflow/pusht/latent_planner_h10.pt}"
LEWM_CHECKPOINT="${LEWM_CHECKPOINT:-$STABLEWM_HOME/lewm_object.ckpt}"
ROLLOUT_CACHE_PATH="${ROLLOUT_CACHE_PATH:-$STABLEWM_HOME/reference_guided_world_model/pusht_h10_staged_rollout_cache.h5}"
SEED="${SEED:-42}"
ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-reference_guided_world_model/pusht_h10}"

conda run -n lewm --no-capture-output python train_reference_guided_world_model.py \
  "data=$DATA" \
  "planner_checkpoint=$PLANNER_CHECKPOINT" \
  "lewm_checkpoint=$LEWM_CHECKPOINT" \
  "rollout_cache.path=$ROLLOUT_CACHE_PATH" \
  "seed=$SEED" \
  "data.world.rollout_workers=$ROLLOUT_WORKERS" \
  "subdir=$OUTPUT_DIR"
