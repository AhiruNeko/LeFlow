#!/bin/bash

mkdir -p logs
FILE_NAME="${BASH_SOURCE[0]##*/}"
TASK_NAME="${FILE_NAME%.*}"
LOG_FILE="logs/${TASK_NAME}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG_FILE") 2>&1

export STABLEWM_HOME=/root/projects/pusht
export HYDRA_FULL_ERROR=1

# Point this at the current bundled, path-only LTC checkpoint. Keeping it as an
# override avoids silently analysing an obsolete delta-z checkpoint after a run.
LTC_CHECKPOINT="${LTC_CHECKPOINT:-${STABLEWM_HOME}/latent_trajectory_cost/pusht_h10_ltc/latent_trajectory_cost_epoch_4.pt}"
ANALYSIS_DIR="${ANALYSIS_DIR:-/root/projects/LeFlow/analysis/latent_trajectory_cost/pusht_h10_path_only}"
mkdir -p "$ANALYSIS_DIR"

conda run -n lewm --no-capture-output python analysis/inspect_ltc_distribution.py \
  --lewm-checkpoint pusht/lewm \
  --ltc-checkpoint "${LTC_CHECKPOINT}" \
  --dataset pusht_expert_train \
  --horizon 10 \
  --action-block 5 \
  --batch-size 64 \
  --num-batches 8 \
  --noise-scales 0,0.02,0.05,0.1,0.2 \
  --output "${ANALYSIS_DIR}/ep4_distribution.json"
