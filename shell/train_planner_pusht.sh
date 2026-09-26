#!/bin/bash

mkdir -p logs
FILE_NAME="${BASH_SOURCE[0]##*/}"
TASK_NAME="${FILE_NAME%.*}"
LOG_FILE="logs/${TASK_NAME}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG_FILE") 2>&1

# Set this to the same checkpoint root used by the completed H=10 LTC run.
export STABLEWM_HOME=/root/projects/pusht
export HYDRA_FULL_ERROR=1

LTC_DIR="${STABLEWM_HOME}/latent_trajectory_cost/pusht_h10_ep10"

conda run -n lewm --no-capture-output python train_latent_planner.py \
  lewm_checkpoint=pusht/lewm \
  data.dataset.name=pusht_expert_train \
  data.dataset.keys_to_load='[pixels,action,proprio,state]' \
  data.dataset.keys_to_cache='[action,proprio,state]' \
  planner.horizon=10 \
  planner.max_horizon=20 \
  planner.action_block=5 \
  flow.path_feature_dim=256 \
  experience.trajectory_encoder_checkpoint="${LTC_DIR}/latent_trajectory_cost_trajectory_encoder_epoch_4.pt" \
  experience.cost_model_checkpoint="${LTC_DIR}/latent_trajectory_cost_cost_model_epoch_4.pt" \
  experience.min_size=0 \
  experience.max_size=64 \
  experience.noise_std=0.10 \
  experience.all_after_start_probability=0.5 \
  epochs=10 \
  loader.batch_size=128 \
  subdir=latent_planner/pusht_phase1_h10_ep10
