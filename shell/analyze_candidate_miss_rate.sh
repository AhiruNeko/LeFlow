#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 {pusht|cube|reacher|tworoom} {5|10|15|20|...}" >&2
  exit 2
fi

TASK="$1"
HORIZON="$2"
if [[ ! "$HORIZON" =~ ^[1-9][0-9]*$ ]]; then
  echo "Horizon must be a positive integer." >&2
  exit 2
fi
cd "$(dirname "$0")/.."
export PYTHONDONTWRITEBYTECODE=1

case "$TASK" in
  pusht)
    PROFILE=pusht
    TASK_HOME=/root/projects/pusht
    PLANNER_REL="latent_planner/pusht_h${HORIZON}/latent_planner.pt"
    GUIDED_REL="reference_guided_world_model/pusht_h${HORIZON}/reference_guided_world_model_best.pt"
    LEWM_REL="checkpoints/pusht/lewm_object.ckpt"
    DATASET_DEFAULT=pusht_expert_train
    ;;
  cube)
    PROFILE=cube
    TASK_HOME=/root/projects/cube
    PLANNER_REL="latent_planner/cube_h${HORIZON}/latent_planner.pt"
    GUIDED_REL="reference_guided_world_model/cube_h${HORIZON}/reference_guided_world_model_best.pt"
    LEWM_REL="cube/cube_object.ckpt"
    DATASET_DEFAULT=ogbench/cube_single_expert
    ;;
  reacher)
    PROFILE=reacher
    TASK_HOME=/root/projects/reacher
    PLANNER_REL="latent_planner/reacher_h${HORIZON}/latent_planner.pt"
    GUIDED_REL="reference_guided_world_model/reacher_h${HORIZON}/reference_guided_world_model_best.pt"
    LEWM_REL="reacher/reacher_object.ckpt"
    DATASET_DEFAULT=/mnt/f/data/reacher
    ;;
  tworoom|tworooms)
    PROFILE=tworoom
    TASK_HOME=/root/projects/tworooms
    PLANNER_REL="leflow/tworooms_h${HORIZON}/latent_planner.pt"
    GUIDED_REL="reference_guided_world_model/tworooms_h${HORIZON}/reference_guided_world_model_best.pt"
    LEWM_REL="tworooms/tworooms_object.ckpt"
    DATASET_DEFAULT=tworoom
    ;;
  *)
    echo "Unknown task: $TASK" >&2
    exit 2
    ;;
esac

export STABLEWM_HOME="${STABLEWM_HOME:-$TASK_HOME}"
PLANNER_CHECKPOINT="${PLANNER_CHECKPOINT:-$STABLEWM_HOME/$PLANNER_REL}"
REFERENCE_GUIDED_CHECKPOINT="${REFERENCE_GUIDED_CHECKPOINT:-$STABLEWM_HOME/$GUIDED_REL}"
LEWM_CHECKPOINT="${LEWM_CHECKPOINT:-$STABLEWM_HOME/$LEWM_REL}"
DATASET_NAME="${DATASET_NAME:-$DATASET_DEFAULT}"
GOAL_OFFSET_STEPS="${GOAL_OFFSET_STEPS:-$((HORIZON * 5))}"
FLOW_STEPS="${FLOW_STEPS:-16}"
HISTORY_SIZE="${HISTORY_SIZE:-3}"
SEED="${SEED:-42}"
TASKS="${TASKS:-500}"
CANDIDATES="${CANDIDATES:-64}"

echo "task=$PROFILE horizon=$HORIZON dataset=$DATASET_NAME"
echo "planner=$PLANNER_CHECKPOINT"
echo "reference_guided=$REFERENCE_GUIDED_CHECKPOINT"
echo "lewm=$LEWM_CHECKPOINT"
echo "sampling $TASKS distinct tasks, $CANDIDATES candidates per task"

conda run -n lewm --no-capture-output python -B analysis/analyze_candidate_miss_rate.py \
  --task "$PROFILE" \
  --dataset-name "$DATASET_NAME" \
  --planner-checkpoint "$PLANNER_CHECKPOINT" \
  --reference-guided-checkpoint "$REFERENCE_GUIDED_CHECKPOINT" \
  --lewm-checkpoint "$LEWM_CHECKPOINT" \
  --expected-horizon "$HORIZON" \
  --goal-offset-steps "$GOAL_OFFSET_STEPS" \
  --flow-steps "$FLOW_STEPS" \
  --history-size "$HISTORY_SIZE" \
  --seed "$SEED" \
  --tasks "$TASKS" \
  --candidates "$CANDIDATES"
