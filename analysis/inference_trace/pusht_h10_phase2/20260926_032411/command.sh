#!/bin/bash

set -euo pipefail

RUN_ID="$(date +%Y%m%d_%H%M%S)"
PROJECT_ROOT="/root/projects/LeFlow"
STABLEWM_HOME="/root/projects/pusht"
MODEL_DIR="$STABLEWM_HOME/latent_planner_finetune/pusht_h10_phase2"
OUTPUT_DIR="$PROJECT_ROOT/analysis/inference_trace/pusht_h10_phase2/$RUN_ID"
LOG_FILE="$OUTPUT_DIR/analyze_inference_trace.log"

mkdir -p "$OUTPUT_DIR"
exec > >(tee "$LOG_FILE") 2>&1

export STABLEWM_HOME
export HYDRA_FULL_ERROR=1

PLANNER="$MODEL_DIR/fine_tuned_latent_planner.pt"
ENCODER="$MODEL_DIR/fine_tuned_latent_planner_trajectory_encoder.pt"
COST="$MODEL_DIR/fine_tuned_latent_planner_cost_model.pt"

for CHECKPOINT in "$PLANNER" "$ENCODER" "$COST"; do
  if [ ! -f "$CHECKPOINT" ]; then
    echo "Missing checkpoint: $CHECKPOINT" >&2
    exit 1
  fi
done

cd "$PROJECT_ROOT"
conda run -n lewm --no-capture-output python analysis/analyze_inference_trace.py \
  --planner-checkpoint "$PLANNER" \
  --encoder-checkpoint "$ENCODER" \
  --cost-checkpoint "$COST" \
  --dataset pusht_expert_train \
  --num-tasks 8 \
  --horizon 10 \
  --action-block 5 \
  --candidates 64 \
  --rounds 1 \
  --experience-max-size 64 \
  --flow-steps 16 \
  --lewm-history-size 3 \
  --batch-size 8 \
  --output-dir "$OUTPUT_DIR"

cp "$BASH_SOURCE" "$OUTPUT_DIR/command.sh"
echo "Artifacts written to: $OUTPUT_DIR"
