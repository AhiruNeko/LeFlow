#!/usr/bin/env bash
set -euo pipefail

export STABLEWM_HOME="${STABLEWM_HOME:-/root/autodl-tmp/cube}"

DATA=ogb
PLANNER_CHECKPOINT="${PLANNER_CHECKPOINT:-$STABLEWM_HOME/leflow/cube_h10_20261016_040459/latent_planner.pt}"
LEWM_CHECKPOINT="${LEWM_CHECKPOINT:-$STABLEWM_HOME/cube/cube_object.ckpt}"
ROLLOUT_CACHE_PATH="${ROLLOUT_CACHE_PATH:-$STABLEWM_HOME/reference_guided_world_model/cube_h10_staged_rollout_cache.h5}"
SEED="${SEED:-42}"
ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-reference_guided_world_model/cube_h10}"

conda run -n lewm --no-capture-output python train_reference_guided_world_model.py \
  "data=$DATA" \
  "planner_checkpoint=$PLANNER_CHECKPOINT" \
  "lewm_checkpoint=$LEWM_CHECKPOINT" \
  "rollout_cache.path=$ROLLOUT_CACHE_PATH" \
  "seed=$SEED" \
  "data.world.rollout_workers=$ROLLOUT_WORKERS" \
  "subdir=$OUTPUT_DIR"
