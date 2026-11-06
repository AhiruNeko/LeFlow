#!/usr/bin/env bash
set -euo pipefail

export STABLEWM_HOME="${STABLEWM_HOME:-/root/autodl-tmp/reacher}"

DATA=dmc
# Default matches shell/train_planner_reacher.sh output; override if needed.
PLANNER_CHECKPOINT="${PLANNER_CHECKPOINT:-$STABLEWM_HOME/leflow/latent_planner/reacher_h10/latent_planner.pt}"
LEWM_CHECKPOINT="${LEWM_CHECKPOINT:-$STABLEWM_HOME/reacher/reacher_object.ckpt}"
ROLLOUT_CACHE_PATH="${ROLLOUT_CACHE_PATH:-$STABLEWM_HOME/reference_guided_world_model/reacher_h10_staged_rollout_cache.h5}"
SEED="${SEED:-42}"
ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-reference_guided_world_model/reacher_h10}"

conda run -n lewm --no-capture-output python train_reference_guided_world_model.py \
  "data=$DATA" \
  "planner_checkpoint=$PLANNER_CHECKPOINT" \
  "lewm_checkpoint=$LEWM_CHECKPOINT" \
  "rollout_cache.path=$ROLLOUT_CACHE_PATH" \
  "seed=$SEED" \
  "data.world.rollout_workers=$ROLLOUT_WORKERS" \
  "staged_training.generate_rollout_cache=true" \
  "staged_training.distillation=true" \
  "staged_training.fine_tune=true" \
  "subdir=$OUTPUT_DIR"
