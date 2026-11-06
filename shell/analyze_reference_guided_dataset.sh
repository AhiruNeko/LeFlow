#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 {pusht|cube|reacher|tworoom} [horizon]" >&2
  exit 2
fi

TASK="$1"
HORIZON="${2:-${HORIZON:-10}}"
cd "$(dirname "$0")/.."

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
    DATASET_DEFAULT=/mnt/f/data/tworoom
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
SAMPLES="${SAMPLES:-500}"
FLOW_STEPS="${FLOW_STEPS:-16}"
HISTORY_SIZE="${HISTORY_SIZE:-3}"
SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-analysis/reference_guided_dataset/${PROFILE}_h${HORIZON}}"

echo "task=$PROFILE horizon=$HORIZON samples=$SAMPLES dataset=$DATASET_NAME"
echo "planner=$PLANNER_CHECKPOINT"
echo "reference_guided=$REFERENCE_GUIDED_CHECKPOINT"
echo "lewm=$LEWM_CHECKPOINT"

conda run -n lewm --no-capture-output python analysis/analyze_reference_guided_dataset.py \
  --task "$PROFILE" \
  --dataset-name "$DATASET_NAME" \
  --planner-checkpoint "$PLANNER_CHECKPOINT" \
  --reference-guided-checkpoint "$REFERENCE_GUIDED_CHECKPOINT" \
  --lewm-checkpoint "$LEWM_CHECKPOINT" \
  --samples "$SAMPLES" \
  --goal-offset-steps "$GOAL_OFFSET_STEPS" \
  --flow-steps "$FLOW_STEPS" \
  --history-size "$HISTORY_SIZE" \
  --seed "$SEED" \
  --output-dir "$OUTPUT_DIR"
