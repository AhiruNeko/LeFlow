#!/usr/bin/env bash
set -euo pipefail

cd /root/projects/LeFlow
export STABLEWM_HOME=/root/projects/pusht
export HYDRA_FULL_ERROR=1

PLANNER_CHECKPOINT=/root/projects/pusht/leflow/pusht/latent_planner_h10.pt
REFERENCE_GUIDED_CHECKPOINT=/root/projects/pusht/reference_guided_world_model/pusht_h10/reference_guided_world_model_best.pt
LEWM_CHECKPOINT=/root/projects/pusht/pusht/lewm_object.ckpt

conda run -n lewm --no-capture-output python analysis/analyze_reference_guided_correlation.py \
  --planner-checkpoint "$PLANNER_CHECKPOINT" \
  --reference-guided-checkpoint "$REFERENCE_GUIDED_CHECKPOINT" \
  --lewm-checkpoint /root/projects/pusht/pusht/lewm_object.ckpt \
  --dataset-name pusht_expert_train \
  --tasks 10 \
  --candidates 64 \
  --goal-offset-steps 50 \
  --flow-steps 16 \
  --history-size 3 \
  --seed 42 \
  --output-dir analysis/reference_guided_correlation
