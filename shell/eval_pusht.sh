#!/bin/bash

mkdir -p logs
FILE_NAME="${BASH_SOURCE[0]##*/}"
TASK_NAME="${FILE_NAME%.*}"
LOG_FILE="logs/${TASK_NAME}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG_FILE") 2>&1

export STABLEWM_HOME=/root/projects/pusht
export HYDRA_FULL_ERROR=1

# Produced by shell/train_planner_pusht.sh.  This checkpoint contains the
# experience-conditioned H=10 flow and inverse-dynamics models.
PLANNER_DIR="${STABLEWM_HOME}/latent_planner/pusht_phase1_h10_ep10"
PLANNER_CHECKPOINT="${PLANNER_DIR}/latent_planner.pt"
LTC_DIR="${STABLEWM_HOME}/latent_trajectory_cost/pusht_h10_ep10"
RESULT_FILE="${TASK_NAME}_results.txt"

conda run -n lewm --no-capture-output python eval.py --config-name=pusht.yaml \
  solver=experience_latent_flow \
  policy="${PLANNER_CHECKPOINT}" \
  plan_config.horizon=10 \
  plan_config.receding_horizon=10 \
  plan_config.action_block=5 \
  solver.trajectory_encoder_checkpoint="${LTC_DIR}/latent_trajectory_cost_trajectory_encoder_epoch_4.pt" \
  solver.cost_model_checkpoint="${LTC_DIR}/latent_trajectory_cost_cost_model_epoch_4.pt" \
  eval.num_eval=50 \
  output.filename="${RESULT_FILE}" \
  eval.goal_offset_steps=50 \
  eval.eval_budget=100

RESULT_DIR="/root/projects/LeFlow/eval_results/${TASK_NAME}"
mkdir -p "${RESULT_DIR}"
if [ -f "${PLANNER_DIR}/${RESULT_FILE}" ]; then
  mv "${PLANNER_DIR}/${RESULT_FILE}" "${RESULT_DIR}/"
fi
