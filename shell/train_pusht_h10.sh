#!/bin/bash

mkdir -p logs
FILE_NAME="${BASH_SOURCE[0]##*/}"
TASK_NAME="${FILE_NAME%.*}"
LOG_FILE="logs/${TASK_NAME}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG_FILE") 2>&1

export STABLEWM_HOME=/root/autodl-tmp/pusht
export HYDRA_FULL_ERROR=1

# Override these environment variables to train against Fast-LeWM.
WORLD_MODEL_TYPE="fast-lewm"
WORLD_MODEL_CHECKPOINT="/root/autodl-tmp/pusht/fast-lewm/pusht/Fast-lewm_pusht_object.ckpt"

conda run -n lewm --no-capture-output python train_latent_planner.py \
  lewm_checkpoint="$WORLD_MODEL_CHECKPOINT" \
  world_model.type="$WORLD_MODEL_TYPE" \
  data.dataset.name=pusht_expert_train \
  data.dataset.keys_to_load='[pixels,action,proprio,state]' \
  data.dataset.keys_to_cache='[action,proprio,state]' \
  planner.horizon=10 \
  planner.max_horizon=20 \
  planner.action_block=5 \
  epochs=10 \
  loader.batch_size=128 \
  subdir=global_path_local_action_flow/pusht_h10_ep10
