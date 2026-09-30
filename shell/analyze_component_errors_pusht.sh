#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
mkdir -p logs
TASK_NAME="${BASH_SOURCE[0]##*/}"
TASK_NAME="${TASK_NAME%.*}"
LOG_FILE="logs/${TASK_NAME}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "${LOG_FILE}") 2>&1

export STABLEWM_HOME=/root/projects/pusht
export HYDRA_FULL_ERROR=1
export SDL_VIDEODRIVER=dummy

PLANNER_CHECKPOINT="${STABLEWM_HOME}/latent_planner_finetune/pusht_h10_phase2/fine_tuned_latent_planner_step_6000.pt"

conda run -n lewm --no-capture-output python analysis/analyze_component_errors.py \
  --planner-checkpoint "${PLANNER_CHECKPOINT}" \
  --dataset-name pusht_expert_train \
  --episode-index 0 \
  --start-step 0 \
  --goal-offset-steps 50 \
  --horizon 10 \
  --action-block 5 \
  --planner-samples 64 \
  --flow-steps 16 \
  --history-size 3 \
  --noise-seed 20260928 \
  --output-dir analysis/component_errors
