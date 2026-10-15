#!/usr/bin/env bash
# Shared lightweight server smoke test. It never contacts WandB and removes
# every run-specific checkpoint, rollout cache, evaluation summary, Hydra run,
# and log before returning.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 {pusht|cube|reacher|tworooms}" >&2
  exit 2
fi

TASK="$1"
PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/LeFlow}"
TASKS_ROOT="${TASKS_ROOT:-/root/autodl-tmp}"
TASK_ROOT="$TASKS_ROOT/$TASK"
[[ -d "$PROJECT_ROOT" ]] || { echo "Missing project root: $PROJECT_ROOT" >&2; exit 1; }

RUN_TAG="smoke_$(date +%Y%m%d_%H%M%S)_$$"
cleanup() {
  rm -rf -- \
    "$TASK_ROOT/leflow/${TASK}_h10_$RUN_TAG" \
    "$TASK_ROOT/reference_guided_world_model/${TASK}_h10_$RUN_TAG" \
    "$TASK_ROOT/eval_results/${TASK}_h10_$RUN_TAG" \
    "$TASK_ROOT/.smoke_runtime/$RUN_TAG" \
    "$PROJECT_ROOT/logs/train_eval_all_${TASK}_$RUN_TAG.log"
}
cleanup_done=false
cleanup_once() {
  if [[ "$cleanup_done" == true ]]; then
    return
  fi
  cleanup_done=true
  cleanup
}
trap cleanup_once EXIT ERR
trap 'cleanup_once; exit 130' INT TERM

export RUN_TAG
export PLANNER_EPOCHS=1
export GUIDED_EPOCHS=1
export PLANNER_BATCH_SIZE=2
export GUIDED_BATCH_SIZE=2
export PLANNER_MAX_TRAIN_BATCHES=1
export GUIDED_MAX_TRAIN_BATCHES=1
export ROLLOUT_CACHE_MAX_RECORDS=4
export ROLLOUT_WORKERS=0
export NUM_EVAL=2
export EVAL_SEEDS="42 43"
export WANDB_MODE=disabled
export WANDB_DISABLED=true
export HYDRA_RUN_DIR="$TASK_ROOT/.smoke_runtime/$RUN_TAG"

cd "$PROJECT_ROOT"
"$PROJECT_ROOT/shell/train_eval_all_${TASK}.sh"

echo "$TASK smoke test completed; all temporary files were removed."