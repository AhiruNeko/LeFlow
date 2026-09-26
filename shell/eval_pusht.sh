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

# Unified planner checkpoint: it embeds frozen IDM and the jointly trained LTC.
PLANNER_DIR="${STABLEWM_HOME}/latent_planner_unified/pusht_h10_ep10"
PLANNER_CHECKPOINT="${PLANNER_DIR}/latent_planner.pt"
RESULT_FILE="${TASK_NAME}_${RUN_ID}_results.txt"
ARTIFACT_MARKER="${PLANNER_DIR}/.${TASK_NAME}_${RUN_ID}.start"
touch "${ARTIFACT_MARKER}"

for CHECKPOINT in "${PLANNER_CHECKPOINT}"; do
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
printf 'planner_checkpoint=%s\n' "${PLANNER_CHECKPOINT}" > "${RESULT_DIR}/checkpoints.txt"

echo "Archived evaluation artifacts to: ${RESULT_DIR}"