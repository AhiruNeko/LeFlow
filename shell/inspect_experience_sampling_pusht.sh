#!/bin/bash
# Visualize raw latent paths sampled by the true candidates x rounds FIFO loop.
# Each experience-bank entry is a prior real flow sample for the same task.

mkdir -p logs
FILE_NAME="${BASH_SOURCE[0]##*/}"
TASK_NAME="${FILE_NAME%.*}"
LOG_FILE="logs/${TASK_NAME}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG_FILE") 2>&1

export STABLEWM_HOME=/root/projects/pusht
export HYDRA_FULL_ERROR=1

PLANNER_DIR="${PLANNER_DIR:-${STABLEWM_HOME}/latent_planner_unified/pusht_h10}"
PLANNER_CHECKPOINT="${PLANNER_CHECKPOINT:-${PLANNER_DIR}/unified_latent_planner.pt}"
# The unified checkpoint contains the path-only trajectory encoder and cost model.
# Override LTC_CHECKPOINT only when inspecting a separately bundled LTC checkpoint.
LTC_CHECKPOINT="${LTC_CHECKPOINT:-${PLANNER_CHECKPOINT}}"
ANALYSIS_DIR="${ANALYSIS_DIR:-/root/projects/LeFlow/analysis/experience_sampling/pusht_h10_path_only}"
mkdir -p "$ANALYSIS_DIR"

conda run -n lewm --no-capture-output python analysis/inspect_experience_sampling.py \
  --planner-checkpoint "${PLANNER_CHECKPOINT}" \
  --ltc-checkpoint "${LTC_CHECKPOINT}" \
  --dataset pusht_expert_train \
  --horizon 10 \
  --action-block 5 \
  --num-tasks 8 \
  --batch-size 8 \
  --candidates-per-round 4 \
  --rounds 17 \
  --experience-max-size 64 \
  --flow-steps 16 \
  --memory-sizes 0,4,16,64 \
  --quality-noise-scales 0.02,0.05,0.1,0.2 \
  --seed 42 \
  --output "${ANALYSIS_DIR}/real_experience_raw_latent_distribution.json"
