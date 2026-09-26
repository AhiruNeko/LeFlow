#!/bin/bash

mkdir -p logs
FILE_NAME="${BASH_SOURCE[0]##*/}"
TASK_NAME="${FILE_NAME%.*}"
LOG_FILE="logs/${TASK_NAME}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG_FILE") 2>&1

export STABLEWM_HOME=/root/projects/pusht
export HYDRA_FULL_ERROR=1

# Smoke test only: two train batches and one validation batch.  Batch size must
# remain >= 2 because the goal-mismatch negative uses another batch element.
conda run -n lewm --no-capture-output python train_ltc.py \
  lewm_checkpoint=pusht/lewm \
  data.dataset.name=pusht_expert_train \
  data.dataset.keys_to_load='[pixels,action,proprio,state]' \
  data.dataset.keys_to_cache='[action,proprio,state]' \
  trajectory.horizon=3 \
  trajectory.action_block=2 \
  trajectory_encoder.max_horizon=3 \
  trajectory_encoder.model_dim=64 \
  trajectory_encoder.representation_dim=64 \
  trajectory_encoder.depth=1 \
  trajectory_encoder.heads=4 \
  trajectory_encoder.mlp_dim=128 \
  loss.sigreg.weight=1.0e-4 \
  loss.sigreg.kwargs.knots=5 \
  loss.sigreg.kwargs.num_proj=16 \
  epochs=1 \
  max_train_batches=2 \
  val_batches=1 \
  loader.batch_size=2 \
  loader.num_workers=0 \
  loader.persistent_workers=false \
  loader.prefetch_factor=null \
  loader.pin_memory=false \
  wandb.enabled=false \
  subdir=latent_trajectory_cost/debug