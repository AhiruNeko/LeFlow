#!/bin/bash
set -euo pipefail

export STABLEWM_HOME="/root/projects/pusht"
export HYDRA_FULL_ERROR=1

PLANNER_CHECKPOINT="/root/projects/pusht/leflow/pusht/latent_planner_h10.pt"
REFERENCE_GUIDED_CHECKPOINT="/root/projects/pusht/reference_guided_world_model/pusht_h10/reference_guided_world_model.pt"
SEEDS=(42 43 44 45 46 47 48 49 50 51)

echo "== Reference-guided LeFlow terminal-only multi-seed evaluation =="
echo "planner=${PLANNER_CHECKPOINT}"
echo "reference_guided=${REFERENCE_GUIDED_CHECKPOINT}"
echo "seeds=${SEEDS[*]}"
echo "horizon=10 receding_horizon=10 action_block=5 goal_offset_steps=50 eval_budget=100"

rates=()
for seed in "${SEEDS[@]}"; do
  echo "===== seed=${seed} ====="
  output=$(conda run -n lewm --no-capture-output python eval.py --config-name=pusht.yaml \
    solver=latent_flow \
    policy="${PLANNER_CHECKPOINT}" \
    solver.reference_guided_checkpoint="${REFERENCE_GUIDED_CHECKPOINT}" \
    plan_config.horizon=10 \
    plan_config.receding_horizon=10 \
    plan_config.action_block=5 \
    eval.goal_offset_steps=50 \
    eval.eval_budget=100 \
    eval.num_eval=50 \
    eval.save_artifacts=false \
    seed="${seed}" 2>&1) || {
      echo "Evaluation failed for seed=${seed}." >&2
      echo "${output}" >&2
      exit 1
    }
  echo "${output}"
  rate=$(printf '%s\n' "${output}" | sed -nE 's/.*evaluation_success_rate=([0-9.]+)%.*/\1/p' | tail -n 1)
  if [[ -z "${rate}" ]]; then
    echo "Could not parse success rate for seed=${seed}." >&2
    exit 1
  fi
  rates+=("${rate}")
done

python3 - "${SEEDS[@]}" -- "${rates[@]}" <<'PY'
import statistics
import sys
separator = sys.argv.index('--')
seeds = [int(value) for value in sys.argv[1:separator]]
rates = [float(value) for value in sys.argv[separator + 1:]]
print('')
print('== Per-seed summary ==')
for seed, rate in zip(seeds, rates):
    print(f'seed={seed} success_rate={rate:.1f}%')
print('')
print('== Aggregate summary ==')
print(f'seeds={len(seeds)} episodes_per_seed=50 total_episodes={len(seeds) * 50}')
print(f'success_rate_mean={statistics.mean(rates):.2f}%')
print(f'success_rate_sample_std={statistics.stdev(rates):.2f}%')
print(f'reported={statistics.mean(rates):.2f} +/- {statistics.stdev(rates):.2f}%')
PY
