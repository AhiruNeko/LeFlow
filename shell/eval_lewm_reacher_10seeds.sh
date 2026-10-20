#!/bin/bash
set -euo pipefail

export STABLEWM_HOME="${STABLEWM_HOME:-/root/autodl-tmp/reacher}"
export HYDRA_FULL_ERROR=1

PLANNER_CHECKPOINT="${PLANNER_CHECKPOINT:-$STABLEWM_HOME/latent_planner/reacher_h10/latent_planner.pt}"
LEWM_CHECKPOINT="${LEWM_CHECKPOINT:-$STABLEWM_HOME/reacher/reacher_object.ckpt}"
SEEDS=(42 43 44 45 46 47 48 49 50 51)
NUM_EVAL=50

echo "== Continuous-LeWM terminal-only multi-seed evaluation =="
echo "planner=${PLANNER_CHECKPOINT}"
echo "lewm=${LEWM_CHECKPOINT}"
echo "reference_guided=disabled (continuous LeWM candidate rollout)"
echo "seeds=${SEEDS[*]}"
echo "horizon=10 receding_horizon=10 action_block=5 goal_offset_steps=50 eval_budget=100"

rates=()
times=()
for seed in "${SEEDS[@]}"; do
  echo "===== seed=${seed} ====="
  output=$(conda run -n lewm --no-capture-output python eval.py --config-name=reacher.yaml \
    solver=latent_flow \
    policy="${PLANNER_CHECKPOINT}" \
    solver.lewm_checkpoint="${LEWM_CHECKPOINT}" \
    plan_config.horizon=10 \
    plan_config.receding_horizon=10 \
    plan_config.action_block=5 \
    eval.goal_offset_steps=50 \
    eval.eval_budget=100 \
    eval.num_eval="${NUM_EVAL}" \
    eval.save_artifacts=false \
    seed="${seed}" 2>&1) || {
      echo "Evaluation failed for seed=${seed}." >&2
      echo "${output}" >&2
      exit 1
    }
  echo "${output}"
  rate=$(printf '%s\n' "${output}" | sed -nE 's/.*evaluation_success_rate=([0-9.]+)%.*/\1/p' | tail -n 1)
  elapsed=$(printf '%s\n' "${output}" | sed -nE 's/.*evaluation_time_seconds=([0-9.]+).*/\1/p' | tail -n 1)
  if [[ -z "${rate}" || -z "${elapsed}" ]]; then
    echo "Could not parse result or evaluation time for seed=${seed}." >&2
    exit 1
  fi
  rates+=("${rate}")
  times+=("${elapsed}")
done

python3 - "${SEEDS[@]}" -- "${rates[@]}" -- "${times[@]}" <<'PY'
import statistics
import sys

first = sys.argv.index("--")
second = sys.argv.index("--", first + 1)
seeds = [int(value) for value in sys.argv[1:first]]
rates = [float(value) for value in sys.argv[first + 1:second]]
times = [float(value) for value in sys.argv[second + 1:]]

if not (len(seeds) == len(rates) == len(times)):
    raise SystemExit("Mismatched seed, score, and timing counts.")

print("")
print("== Per-seed summary ==")
for seed, rate, elapsed in zip(seeds, rates, times):
    print(
        f"seed={seed} success_rate={rate:.1f}% "
        f"evaluation_time_seconds={elapsed:.3f}"
    )

print("")
print("== Aggregate summary ==")
print(f"seeds={len(seeds)} episodes_per_seed=50 total_episodes={len(seeds) * 50}")
print(f"success_rate_mean={statistics.mean(rates):.2f}%")
print(f"success_rate_sample_std={statistics.stdev(rates):.2f}%")
print(f"reported={statistics.mean(rates):.2f} +/- {statistics.stdev(rates):.2f}%")
print(f"evaluation_time_mean_seconds={statistics.mean(times):.3f}")
print(f"evaluation_time_sample_std_seconds={statistics.stdev(times):.3f}")
print(f"evaluation_time_total_seconds={sum(times):.3f}")
PY
