#!/usr/bin/env bash
set -euo pipefail

# Terminal-only fixed 10-seed evaluation.
# The joint checkpoint generates candidates with its training-time world model;
# original LeWM performs the final candidate ranking.
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
SEED_LIST="${SEED_LIST:-42 43 44 45 46 47 48 49 50 51}"
read -r -a SEEDS <<< "$SEED_LIST"

if [[ ${#SEEDS[@]} -eq 0 ]]; then
  echo "SEED_LIST is empty." >&2
  exit 2
fi

echo "== Joint generator + original-LeWM rank: terminal-only multi-seed evaluation =="
echo "policy=$POLICY"
echo "original_lewm_checkpoint=$ORIGINAL_LEWM_CHECKPOINT"
echo "generation_world_model_checkpoint=$GENERATION_WORLD_MODEL_CHECKPOINT"
echo "seeds=${SEEDS[*]}"
echo "horizon=$HORIZON receding_horizon=$RECEDING_HORIZON action_block=$ACTION_BLOCK"
echo "goal_offset_steps=$GOAL_OFFSET_STEPS eval_budget=$EVAL_BUDGET episodes_per_seed=$NUM_EVAL"
echo

scores=()
for seed in "${SEEDS[@]}"; do
  echo "===== seed=$seed ====="
  if ! run_output="$(
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
      solver.history_size="$HISTORY_SIZE" \
      seed="$seed" \
      eval.save_artifacts=false 2>&1
  )"; then
    printf '%s\n' "$run_output"
    echo "Evaluation failed for seed=$seed." >&2
    exit 1
  fi
  printf '%s\n' "$run_output"

  score="$(
    printf '%s\n' "$run_output" |
      sed -n "s/.*'success_rate': \([0-9.]*\).*/\1/p" |
      head -n 1
  )"
  if [[ -z "$score" ]]; then
    echo "Could not parse success_rate for seed=$seed." >&2
    exit 1
  fi
  scores+=("$score")
  printf 'SEED_RESULT seed=%s success_rate=%s%%\n\n' "$seed" "$score"
done

printf '%s\n' "== Per-seed summary =="
for i in "${!SEEDS[@]}"; do
  printf 'seed=%s success_rate=%s%%\n' "${SEEDS[$i]}" "${scores[$i]}"
done

printf '%s\n' "${scores[@]}" | awk -v n_episodes="$NUM_EVAL" '
  { sum += $1; sumsq += $1 * $1; n += 1 }
  END {
    if (n == 0) exit 1
    mean = sum / n
    sd = n > 1 ? sqrt((sumsq - sum * sum / n) / (n - 1)) : 0
    printf "\n== Aggregate summary ==\n"
    printf "seeds=%d episodes_per_seed=%d total_episodes=%d\n", n, n_episodes, n * n_episodes
    printf "success_rate_mean=%.2f%%\n", mean
    printf "success_rate_sample_std=%.2f%%\n", sd
    printf "reported=%.2f +/- %.2f%%\n", mean, sd
  }
'
