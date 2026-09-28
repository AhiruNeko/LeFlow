#!/bin/bash

mkdir -p logs
FILE_NAME="${BASH_SOURCE[0]##*/}"
TASK_NAME="${FILE_NAME%.*}"
LOG_FILE="logs/${TASK_NAME}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG_FILE") 2>&1

export STABLEWM_HOME=/root/autodl-tmp/pusht
export HYDRA_FULL_ERROR=1

PLANNER_DIR="${STABLEWM_HOME}/latent_planner/pusht_phase1_h10_ep10"
PLANNER_CHECKPOINT="${PLANNER_DIR}/latent_planner.pt"

# Closed-loop phase two. Each batch samples candidates uniformly from 1..32
# and runs ceil(64 / candidates) FIFO rounds.
conda run -n lewm --no-capture-output python fine_tuning.py \
  planner_checkpoint="${PLANNER_CHECKPOINT}" \
  data.dataset.name=pusht_expert_train \
  data.dataset.keys_to_load='[pixels,action,proprio,state]' \
  data.dataset.keys_to_cache='[action,proprio,state]' \
  planner.horizon=10 \
  planner.action_block=5 \
  collection.candidates_min=1 \
  collection.candidates_max=32 \
  collection.flow_steps=16 \
  collection.max_size=64 \
  dynamic_ltc.sample_size=8 \
  dynamic_ltc.weight=0.5 \
  loss.experience.weight=0.02 \
  loss.experience.tau=1.0 \
  epochs=3 \
  max_train_batches=10000 \
  validation_interval_steps=1000 \
  val_batches=4 \
  loader.batch_size=32 \
  subdir=latent_planner_finetune/pusht_h10_phase2
