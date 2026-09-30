#!/bin/bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
mkdir -p logs
FILE_NAME="${BASH_SOURCE[0]##*/}"
TASK_NAME="${FILE_NAME%.*}"
LOG_FILE="logs/${TASK_NAME}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "${LOG_FILE}") 2>&1

export STABLEWM_HOME=/root/projects/pusht
export HYDRA_FULL_ERROR=1
export SDL_VIDEODRIVER=dummy

# Planner panels are raw planner latent paths projected to their nearest
# real PushT frame in the same source episode; no inverse dynamics or
# simulator rollout is used for the planner visualization.
PLANNER_DIR="${STABLEWM_HOME}/latent_planner_finetune/pusht_h10_phase2"
PLANNER_CHECKPOINT="${PLANNER_DIR}/fine_tuned_latent_planner_step_6000.pt"

conda run -n lewm --no-capture-output python analysis/visualize_trajectory_paths.py \
  --planner-checkpoint "${PLANNER_CHECKPOINT}" \
  --dataset-name pusht_expert_train \
  --episode-index 0 \
  --start-step 0 \
  --goal-offset-steps 50 \
  --horizon 10 \
  --action-block 5 \
  --flow-steps 16 \
  --candidates 16 \
  --rounds 4 \
  --experience-max-size 64 \
  --num-rendered-paths 4 \
  --noise-seed 20260928 \
  --output-dir analysis/trajectory_rendering
