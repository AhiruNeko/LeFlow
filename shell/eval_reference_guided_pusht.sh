#!/bin/bash
set -euo pipefail

mkdir -p logs
TASK_NAME="${BASH_SOURCE[0]##*/}"
TASK_NAME="${TASK_NAME%.*}"
exec > >(tee "logs/${TASK_NAME}_$(date +%Y%m%d_%H%M%S).log") 2>&1

export STABLEWM_HOME="/root/projects/pusht"
export HYDRA_FULL_ERROR=1

# H=10 reference-guided endpoint evaluation.
SEED=42
PLANNER_CHECKPOINT="/root/projects/pusht/leflow/pusht/latent_planner_h10.pt"
REFERENCE_GUIDED_CHECKPOINT="/root/projects/pusht/reference_guided_world_model/pusht_h10/reference_guided_world_model.pt"

conda run -n lewm --no-capture-output python eval.py --config-name=pusht.yaml \
  solver=latent_flow \
  policy="${PLANNER_CHECKPOINT}" \
  solver.reference_guided_checkpoint="${REFERENCE_GUIDED_CHECKPOINT}" \
  plan_config.horizon=10 \
  plan_config.receding_horizon=10 \
  plan_config.action_block=5 \
  eval.goal_offset_steps=50 \
  eval.eval_budget=100 \
  eval.num_eval=50 \
  seed="${SEED}"

DESTINATION="/root/projects/LeFlow/eval_results/${TASK_NAME}/$(date +%Y%m%d_%H%M%S)"
mkdir -p "${DESTINATION}"
find "${STABLEWM_HOME}/leflow/pusht" -maxdepth 1 -type f \
  \( -name '*.mp4' -o -name '*.gif' -o -name 'pusht_results.txt' \) \
  -exec mv -t "${DESTINATION}" {} +
\n