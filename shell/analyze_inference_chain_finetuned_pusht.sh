#!/bin/bash
set -euo pipefail

FILE_NAME="${BASH_SOURCE[0]##*/}"
TASK_NAME="${FILE_NAME%.*}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RESULT_ROOT="/root/projects/LeFlow/analysis/inference_chain_finetuned"
RUN_LOG_DIR="${RESULT_ROOT}/logs"
mkdir -p "$RUN_LOG_DIR"
LOG_FILE="${RUN_LOG_DIR}/${TASK_NAME}_${RUN_ID}.log"
exec > >(tee "$LOG_FILE") 2>&1

export STABLEWM_HOME=/root/projects/pusht
export HYDRA_FULL_ERROR=1

conda run -n lewm --no-capture-output python analysis/analyze_inference_chain.py   --planner-checkpoint /root/projects/pusht/latent_planner_finetune/pusht_h10_phase2_v1/fine_tuned_latent_planner.pt   --dataset-name pusht_expert_train   --horizon 10   --action-block 5   --goal-offset-steps 50   --flow-steps 16   --history-size 3   --num-tasks 50   --render-tasks 12   --planner-paths 4   --latent-success-threshold 0.10   --seed 42   --output-dir analysis/inference_chain_finetuned
