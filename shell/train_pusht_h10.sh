#!/bin/bash
set -euo pipefail

mkdir -p logs
TASK_NAME="${BASH_SOURCE[0]##*/}"
TASK_NAME="${TASK_NAME%.*}"
exec > >(tee "logs/${TASK_NAME}_$(date +%Y%m%d_%H%M%S).log") 2>&1

export STABLEWM_HOME="${STABLEWM_HOME:-/root/autodl-tmp/pusht}"
export HYDRA_FULL_ERROR=1

# Override this variable to train against another compatible planner checkpoint.
PLANNER_CHECKPOINT="${PLANNER_CHECKPOINT:-/root/autodl-tmp/pusht/leflow/pusht/latent_planner_h10.pt}"

conda run -n lewm --no-capture-output python train_reference_guided_world_model.py \
  planner_checkpoint="$PLANNER_CHECKPOINT" \
  subdir=reference_guided_world_model/pusht_h10
