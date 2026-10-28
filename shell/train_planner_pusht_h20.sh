#!/usr/bin/env bash
set -euo pipefail

export STABLEWM_HOME="${STABLEWM_HOME:-/root/autodl-tmp/pusht}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

LEWM_CHECKPOINT="${LEWM_CHECKPOINT:-$STABLEWM_HOME/checkpoints/pusht/lewm_object.ckpt}"
SEED="${SEED:-42}"
HORIZON="${HORIZON:-20}"
ACTION_BLOCK="${ACTION_BLOCK:-5}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-128}"
OUTPUT_DIR="${OUTPUT_DIR:-latent_planner/pusht_h20}"

conda run -n lewm --no-capture-output python train_latent_planner.py \
  "lewm_checkpoint=$LEWM_CHECKPOINT" \
  "data.dataset.name=pusht_expert_train" \
  "data.dataset.keys_to_load=[pixels,action,proprio,state]" \
  "data.dataset.keys_to_cache=[action,proprio,state]" \
  "planner.horizon=$HORIZON" \
  "planner.max_horizon=20" \
  "planner.action_block=$ACTION_BLOCK" \
  "epochs=$EPOCHS" \
  "loader.batch_size=$BATCH_SIZE" \
  "seed=$SEED" \
  "subdir=$OUTPUT_DIR"
