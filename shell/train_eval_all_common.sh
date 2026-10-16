#!/usr/bin/env bash
# Shared implementation for shell/train_eval_all_{task}.sh.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 {pusht|cube|reacher|tworooms}" >&2
  exit 2
fi

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/LeFlow}"
TASKS_ROOT="${TASKS_ROOT:-/root/autodl-tmp}"
[[ -d "$PROJECT_ROOT" ]] || { echo "Missing project root: $PROJECT_ROOT" >&2; exit 1; }
cd "$PROJECT_ROOT"
source /root/miniconda3/etc/profile.d/conda.sh

TASK="$1"
case "$TASK" in
  pusht)
    TASK_ROOT="$TASKS_ROOT/pusht"
    LEWM_REL=pusht/lewm_object.ckpt
    EVAL_CONFIG=pusht
    DATASET_REL=pusht_expert_train
    PLANNER_DATA_KEYS='pixels,action,state'
    CACHE_KEYS='action,state'
    ;;
  cube)
    TASK_ROOT="$TASKS_ROOT/cube"
    LEWM_REL=cube/cube_object.ckpt
    EVAL_CONFIG=cube
    DATASET_REL=ogbench/cube_single_expert
    PLANNER_DATA_KEYS='pixels,action,observation'
    CACHE_KEYS='action'
    ;;
  reacher)
    TASK_ROOT="$TASKS_ROOT/reacher"
    LEWM_REL=reacher/reacher_object.ckpt
    EVAL_CONFIG=reacher
    DATASET_REL=reacher
    PLANNER_DATA_KEYS='pixels,action,observation'
    CACHE_KEYS='action'
    ;;
  tworooms)
    TASK_ROOT="$TASKS_ROOT/tworooms"
    LEWM_REL=tworooms/tworooms_object.ckpt
    EVAL_CONFIG=tworoom
    DATASET_REL=tworoom
    PLANNER_DATA_KEYS='pixels,action,proprio'
    CACHE_KEYS='action,proprio'
    ;;
esac

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
PLANNER_EPOCHS="${PLANNER_EPOCHS:-10}"
GUIDED_EPOCHS="${GUIDED_EPOCHS:-10}"
NUM_EVAL="${NUM_EVAL:-50}"
ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-16}"
PLANNER_BATCH_SIZE="${PLANNER_BATCH_SIZE:-128}"
GUIDED_BATCH_SIZE="${GUIDED_BATCH_SIZE:-128}"
PLANNER_MAX_TRAIN_BATCHES="${PLANNER_MAX_TRAIN_BATCHES:-}"
GUIDED_MAX_TRAIN_BATCHES="${GUIDED_MAX_TRAIN_BATCHES:-}"
ROLLOUT_CACHE_MAX_RECORDS="${ROLLOUT_CACHE_MAX_RECORDS:-}"
EVAL_SEEDS="${EVAL_SEEDS:-42 43 44 45 46 47 48 49 50 51}"
WANDB_MODE="${WANDB_MODE:-online}"
read -r -a SEEDS <<< "${EVAL_SEEDS}"
HYDRA_ARGS=()
[[ -z "${HYDRA_RUN_DIR:-}" ]] || HYDRA_ARGS+=("hydra.run.dir=$HYDRA_RUN_DIR")

export STABLEWM_HOME="$TASK_ROOT"
export HYDRA_FULL_ERROR=1
export MUJOCO_GL=egl

mkdir -p logs
LOG_NAME="train_eval_all_${TASK}_${RUN_TAG}"
exec > >(tee "logs/${LOG_NAME}.log") 2>&1

LEWM_CHECKPOINT="$TASK_ROOT/$LEWM_REL"
DATASET_PATH="$TASK_ROOT/$DATASET_REL"
PLANNER_SUBDIR="leflow/${TASK}_h10_${RUN_TAG}"
PLANNER_CHECKPOINT="$TASK_ROOT/$PLANNER_SUBDIR/latent_planner.pt"
GUIDED_SUBDIR="reference_guided_world_model/${TASK}_h10_${RUN_TAG}"
GUIDED_CHECKPOINT="$TASK_ROOT/$GUIDED_SUBDIR/reference_guided_world_model_best.pt"
ROLLOUT_CACHE_PATH="$TASK_ROOT/$GUIDED_SUBDIR/${TASK}_h10_rollout_cache.h5"
RESULTS_DIR="$TASK_ROOT/eval_results/${TASK}_h10_${RUN_TAG}"
mkdir -p "$RESULTS_DIR"

PLANNER_DATA_ARGS=(
  "data.dataset.name=$DATASET_PATH"
  ++data.dataset.frameskip=5
  "data.dataset.keys_to_load=[$PLANNER_DATA_KEYS]"
  "data.dataset.keys_to_cache=[$CACHE_KEYS]"
)

echo "== $TASK end-to-end H=10 workflow =="
echo "LeWM: $LEWM_CHECKPOINT"
echo "dataset: $DATASET_PATH.h5"
echo "planner epochs=$PLANNER_EPOCHS; guided epochs=$GUIDED_EPOCHS"
echo "evaluation: H=10, receding H=10, budget=100, goal offset=50"
[[ -f "$LEWM_CHECKPOINT" ]] || { echo "Missing LeWM checkpoint: $LEWM_CHECKPOINT" >&2; exit 1; }
[[ -f "$DATASET_PATH.h5" ]] || { echo "Missing dataset: $DATASET_PATH.h5" >&2; exit 1; }

validate_dataset_columns() {
  local phase="$1"
  local columns="$2"
  conda run -n lewm --no-capture-output python -c '
import h5py
import sys
path, phase, raw_columns = sys.argv[1:]
required = [name for name in raw_columns.split(",") if name]
with h5py.File(path, "r") as handle:
    available = set(handle.keys())
missing = [name for name in required if name not in available]
if missing:
    raise SystemExit(
        f"{phase} requires missing HDF5 fields: {missing}; "
        f"available fields: {sorted(available)}"
    )
' "$DATASET_PATH.h5" "$phase" "$columns"
}

# Fail before model construction when a task dataset schema does not match its
# declared raw inputs. This prevents partial runs and makes test cleanup reliable.
validate_dataset_columns planner "$PLANNER_DATA_KEYS"

PLANNER_LIMIT_ARGS=(loader.batch_size="$PLANNER_BATCH_SIZE")
[[ -z "$PLANNER_MAX_TRAIN_BATCHES" ]] || PLANNER_LIMIT_ARGS+=(max_train_batches="$PLANNER_MAX_TRAIN_BATCHES" val_batches=1)
GUIDED_LIMIT_ARGS=(loader.batch_size="$GUIDED_BATCH_SIZE")
[[ -z "$GUIDED_MAX_TRAIN_BATCHES" ]] || GUIDED_LIMIT_ARGS+=(max_train_batches="$GUIDED_MAX_TRAIN_BATCHES" validation.max_batches=1)
CACHE_LIMIT_ARGS=()
[[ -z "$ROLLOUT_CACHE_MAX_RECORDS" ]] || CACHE_LIMIT_ARGS+=(rollout_cache.max_records="$ROLLOUT_CACHE_MAX_RECORDS")

echo ""
echo "== Stage 1/4: train original LeFlow planner and IDM =="
conda run -n lewm --no-capture-output python train_latent_planner.py \
  lewm_checkpoint="$LEWM_CHECKPOINT" seed=3072 "wandb.mode=$WANDB_MODE" \
  planner.horizon=10 planner.max_horizon=20 planner.action_block=5 \
  epochs="$PLANNER_EPOCHS" output_model_name=latent_planner \
  subdir="$PLANNER_SUBDIR" "${PLANNER_DATA_ARGS[@]}" "${PLANNER_LIMIT_ARGS[@]}" "${HYDRA_ARGS[@]}"

[[ -f "$PLANNER_CHECKPOINT" ]] || { echo "Missing planner checkpoint: $PLANNER_CHECKPOINT" >&2; exit 1; }

echo ""
echo "== Stage 2/4: collect real rollouts and train reference-guided model =="
conda run -n lewm --no-capture-output python train_reference_guided_world_model.py \
  planner_checkpoint="$PLANNER_CHECKPOINT" lewm_checkpoint="$LEWM_CHECKPOINT" \
  seed=42 epochs="$GUIDED_EPOCHS" subdir="$GUIDED_SUBDIR" "wandb.mode=$WANDB_MODE" \
  "data=$TASK" \
  rollout_cache.mode=generate rollout_cache.path="$ROLLOUT_CACHE_PATH" \
  rollout_cache.collection_seed=42 rollout_cache.overwrite=false \
  rollout_cache.generate_only=false data.world.rollout_workers="$ROLLOUT_WORKERS" \
  "${GUIDED_LIMIT_ARGS[@]}" "${CACHE_LIMIT_ARGS[@]}" "${HYDRA_ARGS[@]}"

[[ -f "$GUIDED_CHECKPOINT" ]] || { echo "Missing reference-guided checkpoint: $GUIDED_CHECKPOINT" >&2; exit 1; }

run_eval() {
  local label="$1"
  local guided="$2"
  local table="$RESULTS_DIR/${label}_10seeds.tsv"
  printf 'seed\tsuccess_rate\tevaluation_time_seconds\n' > "$table"

  for seed in "${SEEDS[@]}"; do
    echo "===== $label seed=$seed ====="
    local cmd=(
      conda run -n lewm --no-capture-output python eval.py
      "--config-name=${EVAL_CONFIG}.yaml" solver=latent_flow
      "policy=$PLANNER_CHECKPOINT"
      "eval.dataset_name=$DATASET_PATH"
      "dataset.stats=$DATASET_PATH"
      plan_config.horizon=10 plan_config.receding_horizon=10 plan_config.action_block=5
      eval.goal_offset_steps=50 eval.eval_budget=100 "eval.num_eval=$NUM_EVAL"
      ++eval.save_artifacts=false "seed=$seed" "${HYDRA_ARGS[@]}"
    )
    [[ -z "$guided" ]] || cmd+=("solver.reference_guided_checkpoint=$guided")

    local output rate elapsed
    output=$("${cmd[@]}" 2>&1) || { echo "$output" >&2; exit 1; }
    echo "$output"
    rate=$(printf '%s\n' "$output" | sed -nE 's/.*evaluation_success_rate=([0-9.]+)%.*/\1/p' | tail -n 1)
    elapsed=$(printf '%s\n' "$output" | sed -nE 's/.*evaluation_time_seconds=([0-9.]+).*/\1/p' | tail -n 1)
    [[ -n "$rate" && -n "$elapsed" ]] || { echo "Could not parse $label seed=$seed." >&2; exit 1; }
    printf '%s\t%s\t%s\n' "$seed" "$rate" "$elapsed" >> "$table"
  done

  local summary="$RESULTS_DIR/${label}_10seeds_summary.txt"
  python3 - "$label" "$table" "$NUM_EVAL" <<'PY' | tee "$summary"
import csv, statistics, sys
label, path, episodes = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(path, newline="") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))
rates = [float(r["success_rate"]) for r in rows]
times = [float(r["evaluation_time_seconds"]) for r in rows]
print(f"\n== {label} aggregate summary ==")
for r in rows:
    print(f"seed={r['seed']} success_rate={float(r['success_rate']):.1f}% evaluation_time_seconds={float(r['evaluation_time_seconds']):.3f}")
print(f"seeds={len(rows)} episodes_per_seed={episodes} total_episodes={len(rows) * episodes}")
print(f"success_rate_mean={statistics.mean(rates):.2f}%")
print(f"success_rate_sample_std={statistics.stdev(rates):.2f}%")
print(f"reported={statistics.mean(rates):.2f} +/- {statistics.stdev(rates):.2f}%")
print(f"evaluation_time_mean_seconds={statistics.mean(times):.3f}")
print(f"evaluation_time_sample_std_seconds={statistics.stdev(times):.3f}")
print(f"evaluation_time_total_seconds={sum(times):.3f}")
PY
  # The compact summary above already includes every seed; remove the temporary TSV.
  rm -f -- "$table"
}

echo ""
echo "== Stage 3/4: original LeFlow 10-seed evaluation =="
run_eval original_leflow ""
echo ""
echo "== Stage 4/4: reference-guided LeFlow 10-seed evaluation =="
run_eval reference_guided_leflow "$GUIDED_CHECKPOINT"

echo ""
echo "Completed $TASK workflow."
echo "planner_checkpoint=$PLANNER_CHECKPOINT"
echo "reference_guided_checkpoint=$GUIDED_CHECKPOINT"
echo "results_dir=$RESULTS_DIR"
