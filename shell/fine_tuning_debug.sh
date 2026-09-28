#!/bin/bash

mkdir -p logs
FILE_NAME="${BASH_SOURCE[0]##*/}"
TASK_NAME="${FILE_NAME%.*}"
LOG_FILE="logs/${TASK_NAME}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG_FILE") 2>&1

export STABLEWM_HOME=/root/projects/pusht
export HYDRA_FULL_ERROR=1

PLANNER_DIR="${STABLEWM_HOME}/latent_planner/pusht_phase1_h10_ep10"
PLANNER_CHECKPOINT="${PLANNER_DIR}/latent_planner.pt"

# Smoke test: preserve bundled-checkpoint loading while minimizing collection,
# rollout labels, validation, data loading, and parameter updates.
conda run -n lewm --no-capture-output python fine_tuning.py \
  planner_checkpoint="${PLANNER_CHECKPOINT}" \
  data.dataset.name=pusht_expert_train \
  data.dataset.keys_to_load='[pixels,action,proprio,state]' \
  data.dataset.keys_to_cache='[action,proprio,state]' \
  planner.horizon=10 \
  planner.action_block=5 \
  collection.candidates_min=1 \
  collection.candidates_max=2 \
  collection.flow_steps=1 \
  collection.max_size=2 \
  dynamic_ltc.sample_size=2 \
  loss.experience.weight=0.02 \
  loss.experience.tau=1.0 \
  epochs=1 \
  max_train_batches=1 \
  validation_interval_steps=1 \
  val_batches=1 \
  loader.batch_size=2 \
  loader.num_workers=0 \
  loader.persistent_workers=false \
  loader.prefetch_factor=null \
  loader.pin_memory=false \
  wandb.enabled=false \
  subdir=latent_planner_finetune/debug
