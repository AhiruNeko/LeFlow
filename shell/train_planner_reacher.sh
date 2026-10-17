#!/usr/bin/env bash
set -euo pipefail

export STABLEWM_HOME="${STABLEWM_HOME:-/root/autodl-tmp/reacher}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

LEWM_CHECKPOINT="${LEWM_CHECKPOINT:-$STABLEWM_HOME/reacher/reacher_object.ckpt}"
SEED="${SEED:-42}"
HORIZON="${HORIZON:-10}"
ACTION_BLOCK="${ACTION_BLOCK:-5}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-128}"
OUTPUT_DIR="${OUTPUT_DIR:-latent_planner/reacher_h10}"

conda run -n lewm --no-capture-output python train_latent_planner.py \
  "lewm_checkpoint=$LEWM_CHECKPOINT" \
  "data.dataset.name=reacher" \
  "data.dataset.keys_to_load=[pixels,action,observation]" \
  "data.dataset.keys_to_cache=[action,observation]" \
  "planner.horizon=$HORIZON" \
  "planner.max_horizon=20" \
  "planner.action_block=$ACTION_BLOCK" \
  "epochs=$EPOCHS" \
  "loader.batch_size=$BATCH_SIZE" \
  "seed=$SEED" \
  "subdir=$OUTPUT_DIR"
