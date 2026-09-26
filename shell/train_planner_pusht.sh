#!/bin/bash

mkdir -p logs
FILE_NAME="${BASH_SOURCE[0]##*/}"
TASK_NAME="${FILE_NAME%.*}"
LOG_FILE="logs/${TASK_NAME}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG_FILE") 2>&1

export STABLEWM_HOME=/root/autodl-tmp/pusht
export HYDRA_FULL_ERROR=1

# Train this path-only, sigmoid-bounded LTC checkpoint before launching.
LTC_DIR="${STABLEWM_HOME}/latent_trajectory_cost/pusht_h10_ltc"
ORIGINAL_PLANNER="${STABLEWM_HOME}/leflow/pusht/latent_planner.pt"

conda run -n lewm --no-capture-output python train_latent_planner.py \
  lewm_checkpoint=pusht/lewm \
  inverse_dynamics_checkpoint="$ORIGINAL_PLANNER" \
  data.dataset.name=pusht_expert_train \
  data.dataset.keys_to_load='[pixels,action,proprio,state]' \
  data.dataset.keys_to_cache='[action,proprio,state]' \
  planner.horizon=10 \
  planner.max_horizon=20 \
  planner.action_block=5 \
  flow.path_feature_dim=256 \
  experience.checkpoint="${LTC_DIR}/latent_trajectory_cost_epoch_4.pt" \
  training.bootstrap_synthetic_updates=500 \
  training.epochs=10 \
  training.cycles_per_epoch=4 \
  training.real_updates_per_cycle=25 \
  collection.task_batch_size=8 \
  collection.candidates_choices='[4,8,16]' \
  collection.rounds_choices='[1,2,4]' \
  collection.flow_steps=16 \
  synthetic.memory_min_size=0 \
  synthetic.memory_max_size=64 \
  real_replay.max_memory_size=64 \
  real_replay.cache_max_banks=16 \
  loader.batch_size=128 \
  subdir=latent_planner_unified/pusht_h10_ep10
