#!/bin/bash

set -euo pipefail

FILE_NAME="${BASH_SOURCE[0]##*/}"
TASK_NAME="${FILE_NAME%.*}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="/root/projects/LeFlow/eval_results/${TASK_NAME}/${RUN_ID}"
mkdir -p "${RESULT_DIR}"

# Keep the terminal log with the corresponding videos and result text.
LOG_FILE="${RESULT_DIR}/${TASK_NAME}.log"
exec > >(tee "${LOG_FILE}") 2>&1

export STABLEWM_HOME=/root/projects/pusht
export HYDRA_FULL_ERROR=1

# Phase-two fine-tuned planner and jointly fine-tuned LTC components.
PLANNER_DIR="${STABLEWM_HOME}/latent_planner_finetune/pusht_h10_phase2"
PLANNER_CHECKPOINT="${PLANNER_DIR}/fine_tuned_latent_planner.pt"
TRAJECTORY_ENCODER_CHECKPOINT="${PLANNER_DIR}/fine_tuned_latent_planner_trajectory_encoder.pt"
COST_MODEL_CHECKPOINT="${PLANNER_DIR}/fine_tuned_latent_planner_cost_model.pt"
RESULT_FILE="${TASK_NAME}_${RUN_ID}_results.txt"
ARTIFACT_MARKER="${PLANNER_DIR}/.${TASK_NAME}_${RUN_ID}.start"
touch "${ARTIFACT_MARKER}"

for CHECKPOINT in \
  "${PLANNER_CHECKPOINT}" \
  "${TRAJECTORY_ENCODER_CHECKPOINT}" \
  "${COST_MODEL_CHECKPOINT}"; do
  if [ ! -f "${CHECKPOINT}" ]; then
    echo "Missing checkpoint: ${CHECKPOINT}" >&2
    exit 1
  fi
done

conda run -n lewm --no-capture-output python eval.py --config-name=pusht.yaml \
  solver=experience_latent_flow \
  policy="${PLANNER_CHECKPOINT}" \
  plan_config.horizon=10 \
  plan_config.receding_horizon=10 \
  plan_config.action_block=5 \
  solver.trajectory_encoder_checkpoint="${TRAJECTORY_ENCODER_CHECKPOINT}" \
  solver.cost_model_checkpoint="${COST_MODEL_CHECKPOINT}" \
  eval.num_eval=50 \
  output.filename="${RESULT_FILE}"

# eval.py emits its text and rollout videos beside the policy checkpoint.
# Archive only this run's uniquely named text file and its rollout artifacts;
# the model weights remain in PLANNER_DIR.
shopt -s nullglob
ARTIFACTS=("${PLANNER_DIR}/${RESULT_FILE}")
while IFS= read -r -d $'\0' artifact; do
  ARTIFACTS+=("${artifact}")
done < <(find "${PLANNER_DIR}" -maxdepth 1 -type f -name 'rollout_*.mp4' -newer "${ARTIFACT_MARKER}" -print0)
if [ "${#ARTIFACTS[@]}" -gt 0 ]; then
  mv "${ARTIFACTS[@]}" "${RESULT_DIR}/"
fi
rm -f "${ARTIFACT_MARKER}"

cp "${BASH_SOURCE[0]}" "${RESULT_DIR}/command.sh"
printf 'planner_checkpoint=%s\ntrajectory_encoder_checkpoint=%s\ncost_model_checkpoint=%s\n' \
  "${PLANNER_CHECKPOINT}" \
  "${TRAJECTORY_ENCODER_CHECKPOINT}" \
  "${COST_MODEL_CHECKPOINT}" \
  > "${RESULT_DIR}/checkpoints.txt"

echo "Archived evaluation artifacts to: ${RESULT_DIR}"