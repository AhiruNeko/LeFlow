#!/usr/bin/env bash
set -euo pipefail

# Generate candidates with the joint flow checkpoint, then rank them with
# original autoregressive LeWM. Override any variable from the command line.
export STABLEWM_HOME="${STABLEWM_HOME:-/root/projects/pusht}"
export HYDRA_FULL_ERROR=1

POLICY="${POLICY:-global_path_local_action_flow/pusht_h5_ep10/global_path_local_action_flow.pt}"
ORIGINAL_LEWM_CHECKPOINT="${ORIGINAL_LEWM_CHECKPOINT:-pusht/lewm}"
GENERATION_WORLD_MODEL_CHECKPOINT="${GENERATION_WORLD_MODEL_CHECKPOINT:-/root/projects/pusht/fast-lewm/pusht/Fast-lewm_pusht_object.ckpt}"
HORIZON="${HORIZON:-5}"
RECEDING_HORIZON="${RECEDING_HORIZON:-5}"
ACTION_BLOCK="${ACTION_BLOCK:-5}"
GOAL_OFFSET_STEPS="${GOAL_OFFSET_STEPS:-25}"
EVAL_BUDGET="${EVAL_BUDGET:-50}"
NUM_EVAL="${NUM_EVAL:-50}"
NUM_SAMPLES="${NUM_SAMPLES:-64}"
FLOW_STEPS="${FLOW_STEPS:-16}"
HISTORY_SIZE="${HISTORY_SIZE:-3}"

conda run -n lewm --no-capture-output python eval_original_lewm_rank.py --config-name=pusht.yaml \
  solver=original_lewm_rank_joint \
  policy="$POLICY" \
  solver.original_lewm_checkpoint="$ORIGINAL_LEWM_CHECKPOINT" \
      solver.world_model_checkpoint="$GENERATION_WORLD_MODEL_CHECKPOINT" \
  plan_config.horizon="$HORIZON" \
  plan_config.receding_horizon="$RECEDING_HORIZON" \
  plan_config.action_block="$ACTION_BLOCK" \
  eval.eval_budget="$EVAL_BUDGET" \
  eval.goal_offset_steps="$GOAL_OFFSET_STEPS" \
  eval.num_eval="$NUM_EVAL" \
  solver.num_samples="$NUM_SAMPLES" \
  solver.flow_steps="$FLOW_STEPS" \
  solver.history_size="$HISTORY_SIZE"
