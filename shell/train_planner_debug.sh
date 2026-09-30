#!/bin/bash

mkdir -p logs
FILE_NAME="${BASH_SOURCE[0]##*/}"
TASK_NAME="${FILE_NAME%.*}"
LOG_FILE="logs/${TASK_NAME}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG_FILE") 2>&1

# Smoke test only: two train batches, one validation batch, and a small flow.
export STABLEWM_HOME=/root/projects/pusht
export HYDRA_FULL_ERROR=1

LTC_DIR="${STABLEWM_HOME}/latent_trajectory_cost/pusht_h10_ep10"

conda run -n lewm --no-capture-output python train_latent_planner.py \
  lewm_checkpoint=pusht/lewm \
  data.dataset.name=pusht_expert_train \
  data.dataset.keys_to_load='[pixels,action,proprio,state]' \
  data.dataset.keys_to_cache='[action,proprio,state]' \
  planner.horizon=3 \
  planner.max_horizon=4 \
  planner.action_block=5 \
  flow.hidden_dim=64 \
  flow.depth=1 \
  flow.time_dim=32 \
  flow.heads=4 \
  flow.path_feature_dim=256 \
  inverse_dynamics.hidden_dim=64 \
  inverse_dynamics.depth=1 \
  experience.trajectory_encoder_checkpoint="${LTC_DIR}/latent_trajectory_cost_trajectory_encoder_epoch_4.pt" \
  experience.cost_model_checkpoint="${LTC_DIR}/latent_trajectory_cost_cost_model_epoch_4.pt" \
  experience.min_size=0 \
  experience.max_size=2 \
  epochs=1 \
  max_train_batches=2 \
  val_batches=1 \
  loader.batch_size=2 \
  loader.num_workers=0 \
  loader.persistent_workers=false \
  loader.prefetch_factor=null \
  loader.pin_memory=false \
  wandb.enabled=false \
  subdir=latent_planner/debug_phase1
