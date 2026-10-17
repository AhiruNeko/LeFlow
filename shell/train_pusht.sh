#!/bin/bash
set -euo pipefail

mkdir -p logs
TASK_NAME="${BASH_SOURCE[0]##*/}"
TASK_NAME="${TASK_NAME%.*}"
exec > >(tee "logs/${TASK_NAME}_$(date +%Y%m%d_%H%M%S).log") 2>&1

export STABLEWM_HOME="/root/autodl-tmp/pusht"
export HYDRA_FULL_ERROR=1

# Reproducible frozen-data collection, followed immediately by offline training.
TRAIN_SEED=42
COLLECTION_SEED=42
ROLLOUT_WORKERS=16
PLANNER_CHECKPOINT="/root/autodl-tmp/pusht/leflow/pusht/latent_planner.pt"
LEWM_CHECKPOINT="/root/autodl-tmp/pusht/lewm_object.ckpt"
ROLLOUT_CACHE_PATH="/root/autodl-tmp/pusht/reference_guided_world_model/pusht_h5_rollout_cache.h5"

conda run -n lewm --no-capture-output python train_reference_guided_world_model.py \
  seed="${TRAIN_SEED}" \
  planner_checkpoint="${PLANNER_CHECKPOINT}" \
  lewm_checkpoint="${LEWM_CHECKPOINT}" \
  rollout_cache.mode=generate \
  rollout_cache.path="${ROLLOUT_CACHE_PATH}" \
  rollout_cache.overwrite=true \
  rollout_cache.generate_only=false \
  rollout_cache.collection_seed="${COLLECTION_SEED}" \
  world.rollout_workers="${ROLLOUT_WORKERS}" \
  subdir=reference_guided_world_model/pusht_h5
