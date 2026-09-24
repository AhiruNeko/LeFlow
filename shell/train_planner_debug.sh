#!/bin/bash

mkdir -p logs
FILE_NAME="${BASH_SOURCE[0]##*/}"
TASK_NAME="${FILE_NAME%.*}"
LOG_FILE="logs/${TASK_NAME}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG_FILE") 2>&1

export STABLEWM_HOME=/root/projects/pusht
export HYDRA_FULL_ERROR=1

# Requires a newly trained state-plus-goal-delta, sigmoid-bounded H=10 LTC checkpoint.
LTC_DIR="${STABLEWM_HOME}/latent_trajectory_cost/pusht_h10_delta_sigmoid"
ORIGINAL_PLANNER="${STABLEWM_HOME}/leflow/pusht/latent_planner_h10.pt"

conda run -n lewm --no-capture-output python train_latent_planner.py \
  lewm_checkpoint=pusht/lewm \
  inverse_dynamics_checkpoint="$ORIGINAL_PLANNER" \
  data.dataset.name=pusht_expert_train \
  data.dataset.keys_to_load='[pixels,action,proprio,state]' \
  data.dataset.keys_to_cache='[action,proprio,state]' \
  planner.horizon=10 \
  planner.max_horizon=20 \
  planner.action_block=5 \
  flow.hidden_dim=64 \
  flow.depth=1 \
  flow.time_dim=32 \
  flow.heads=4 \
  flow.path_feature_dim=256 \
  experience.checkpoint="${LTC_DIR}/latent_trajectory_cost.pt" \
  training.bootstrap_synthetic_updates=2 \
  training.epochs=1 \
  training.cycles_per_epoch=1 \
  training.real_updates_per_cycle=2 \
  collection.task_batch_size=2 \
  collection.candidates_choices='[2]' \
  collection.rounds_choices='[1]' \
  collection.flow_steps=2 \
  collection.max_paths_per_task=2 \
  real_replay.cache_max_banks=1 \
  real_replay.memory_size_choices='[0,1,2]' \
  loader.batch_size=4 \
  loader.num_workers=0 \
  loader.persistent_workers=false \
  loader.prefetch_factor=null \
  loader.pin_memory=false \
  wandb.enabled=false \
  subdir=latent_planner_unified/debug_h10
