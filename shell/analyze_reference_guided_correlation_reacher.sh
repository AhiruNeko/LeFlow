#!/usr/bin/env bash
set -euo pipefail

cd /root/projects/LeFlow
export STABLEWM_HOME="${STABLEWM_HOME:-/root/autodl-tmp/reacher}"
export HYDRA_FULL_ERROR=1

PLANNER_CHECKPOINT="${PLANNER_CHECKPOINT:-$STABLEWM_HOME/latent_planner/reacher_h10/latent_planner.pt}"
REFERENCE_GUIDED_CHECKPOINT="${REFERENCE_GUIDED_CHECKPOINT:-$STABLEWM_HOME/reference_guided_world_model/reacher_h10/reference_guided_world_model_best.pt}"
LEWM_CHECKPOINT="${LEWM_CHECKPOINT:-$STABLEWM_HOME/reacher/reacher_object.ckpt}"
OUTPUT_DIR="${OUTPUT_DIR:-analysis/reference_guided_correlation/reacher_h10}"

conda run -n lewm --no-capture-output python analysis/analyze_reference_guided_correlation.py \
  --task reacher \
  --planner-checkpoint "$PLANNER_CHECKPOINT" \
  --reference-guided-checkpoint "$REFERENCE_GUIDED_CHECKPOINT" \
  --lewm-checkpoint "$LEWM_CHECKPOINT" \
  --dataset-name dmc/reacher \
  --tasks 10 \
  --candidates 64 \
  --goal-offset-steps 50 \
  --flow-steps 16 \
  --history-size 3 \
  --seed 42 \
  --output-dir "$OUTPUT_DIR"
