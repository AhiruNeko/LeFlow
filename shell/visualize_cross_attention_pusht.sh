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

conda run -n lewm --no-capture-output python analysis/visualize_cross_attention.py \
  --planner-checkpoint "${PLANNER_CHECKPOINT}" \
  --dataset-name pusht_expert_train \
  --episode-index 0 \
  --start-step 0 \
  --goal-offset-steps 50 \
  --horizon 10 \
  --action-block 5 \
  --flow-steps 16 \
  --candidates 8 \
  --rounds 8 \
  --max-memory-size 64 \
  --noise-seed 2026 \
  --output-dir analysis/cross_attention
