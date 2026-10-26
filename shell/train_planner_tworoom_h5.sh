#!/usr/bin/env bash
set -euo pipefail

export STABLEWM_HOME="${STABLEWM_HOME:-/root/autodl-tmp/tworooms}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

LEWM_CHECKPOINT="${LEWM_CHECKPOINT:-$STABLEWM_HOME/tworooms/tworooms_object.ckpt}"
SEED="${SEED:-42}"
HORIZON="${HORIZON:-5}"
ACTION_BLOCK="${ACTION_BLOCK:-5}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-128}"
OUTPUT_DIR="${OUTPUT_DIR:-latent_planner/tworoom_h5}"

conda run -n lewm --no-capture-output python train_latent_planner.py \
  "lewm_checkpoint=$LEWM_CHECKPOINT" \
  "data.dataset.name=tworoom" \
  "data.dataset.keys_to_load=[pixels,action,proprio]" \
  "data.dataset.keys_to_cache=[action,proprio]" \
  "planner.horizon=$HORIZON" \
  "planner.max_horizon=20" \
  "planner.action_block=$ACTION_BLOCK" \
  "epochs=$EPOCHS" \
  "loader.batch_size=$BATCH_SIZE" \
  "seed=$SEED" \
  "subdir=$OUTPUT_DIR"
