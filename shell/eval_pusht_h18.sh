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
  policy=leflow/pusht/latent_planner.pt \
  plan_config.horizon=18 \
  plan_config.receding_horizon=18 \
  plan_config.action_block=5 \
  eval.num_eval=50 \
  eval.eval_budget=100

mkdir -p /root/projects/LeFlow/eval_results/$TASK_NAME

find $STABLEWM_HOME/leflow/pusht -maxdepth 1 -type f \
  -not -name "latent_planner.pt" \
  -not -name "latent_planner_config.yaml" \
  -exec mv -t /root/projects/LeFlow/eval_results/$TASK_NAME {} +
