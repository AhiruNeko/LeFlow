#!/bin/bash

mkdir -p logs
FILE_NAME="${BASH_SOURCE[0]##*/}"
TASK_NAME="${FILE_NAME%.*}"
LOG_FILE="logs/${TASK_NAME}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG_FILE") 2>&1

export STABLEWM_HOME=/root/projects/pusht
export HYDRA_FULL_ERROR=1

conda run -n lewm --no-capture-output python eval.py --config-name=pusht.yaml \
  solver=latent_flow \
  policy=global_path_local_action_flow/pusht_h5_ep10/global_path_local_action_flow.pt \
  solver.world_model_checkpoint=/root/projects/pusht/fast-lewm/pusht/Fast-lewm_pusht_object.ckpt \
  plan_config.horizon=5 \
  plan_config.receding_horizon=5 \
  plan_config.action_block=5 \
  eval.eval_budget=50 \
  eval.goal_offset_steps=25 \
  eval.num_eval=50

RESULT_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULT_DIR="/root/projects/LeFlow/eval_results/$TASK_NAME/$RESULT_TIMESTAMP"
mkdir -p "$RESULT_DIR"

find $STABLEWM_HOME/global_path_local_action_flow/pusht_h5_ep10 -maxdepth 1 -type f -not -name "global_path_local_action_flow.pt" -not -name "latent_planner_config.yaml" -exec mv -t "$RESULT_DIR" {} +

echo "Evaluation artifacts saved to: $RESULT_DIR"
