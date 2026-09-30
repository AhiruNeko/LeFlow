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

# The checkpoint must bundle planner, inverse dynamics, and experience.ltc.
# Every schedule receives one fixed task and the same 64 initial noise paths.
PLANNER_DIR="${STABLEWM_HOME}/latent_planner_finetune/pusht_h10_phase2"
PLANNER_CHECKPOINT="${PLANNER_DIR}/fine_tuned_latent_planner_step_6000.pt"

conda run -n lewm --no-capture-output python analysis/analyze_candidate_rollouts.py \
  --planner-checkpoint "${PLANNER_CHECKPOINT}" \
  --dataset-name pusht_expert_train \
  --episode-index 0 \
  --start-step 0 \
  --goal-offset-steps 50 \
  --horizon 10 \
  --action-block 5 \
  --flow-steps 16 \
  --history-size 3 \
  --max-paths 64 \
  --schedules 64x1,32x2,16x4,8x8,4x16 \
  --noise-seed 63456 \
  --output-dir analysis/candidate_rollout_comparison
