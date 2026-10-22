#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
VARIANT="${1:?Expected no_local or endpoints_actions}"
shift
case "$VARIANT" in no_local|endpoints_actions) ;; *) echo "Unknown ablation: $VARIANT" >&2; exit 2 ;; esac
cd "$REPO_ROOT"
export PYTHONDONTWRITEBYTECODE=1
export HYDRA_FULL_ERROR=1
export STABLEWM_HOME="${STABLEWM_HOME:-/root/autodl-tmp/pusht}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/ablation/outputs/.matplotlib}"

PLANNER_CHECKPOINT="${PLANNER_CHECKPOINT:-$STABLEWM_HOME/leflow/pusht/latent_planner_h10.pt}"
LEWM_CHECKPOINT="${LEWM_CHECKPOINT:-$STABLEWM_HOME/pusht/lewm_object.ckpt}"
REFERENCE_GUIDED_CHECKPOINT="${REFERENCE_GUIDED_CHECKPOINT:-$STABLEWM_HOME/reference_guided_world_model/pusht_h10_fine_tune/reference_guided_world_model_best.pt}"
ROLLOUT_CACHE_PATH="${ROLLOUT_CACHE_PATH:-$STABLEWM_HOME/reference_guided_world_model/pusht_h10_staged_rollout_cache.h5}"
DATASET_NAME="${DATASET_NAME:-pusht_expert_train}"
SEED="${SEED:-42}"
NUM_EVAL="${NUM_EVAL:-50}"
ANALYSIS_TASKS="${ANALYSIS_TASKS:-10}"
ANALYSIS_CANDIDATES="${ANALYSIS_CANDIDATES:-64}"
DISTILLATION_EPOCHS="${DISTILLATION_EPOCHS:-10}"
FINE_TUNE_EPOCHS="${FINE_TUNE_EPOCHS:-25}"
DISTILLATION_LR="${DISTILLATION_LR:-1e-4}"
FINE_TUNE_LR="${FINE_TUNE_LR:-3e-5}"
LOADER_WORKERS="${LOADER_WORKERS:-6}"
BATCH_SIZE="${BATCH_SIZE:-128}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/ablation/outputs/pusht/$VARIANT/$RUN_TAG}"
OUTPUT_DIR="$(realpath -m -- "$OUTPUT_DIR")"
case "$OUTPUT_DIR/" in "$REPO_ROOT/ablation/"*) ;; *) echo "OUTPUT_DIR must be inside $REPO_ROOT/ablation" >&2; exit 2 ;; esac
ABLATION_CHECKPOINT="$OUTPUT_DIR/train/ablation_${VARIANT}_best.pt"
read -r -a SEEDS <<< "${EVAL_SEEDS:-42 43 44 45 46 47 48 49 50 51}"
if [[ "${#SEEDS[@]}" -ne 10 ]]; then echo "EVAL_SEEDS must contain exactly ten seeds." >&2; exit 2; fi
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON=("$PYTHON_BIN" -B)
else
  PYTHON=(conda run -n "${CONDA_ENV:-lewm}" --no-capture-output python -B)
fi

print_command() { printf '%q ' "$@"; printf '\n'; }
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  for file in "$PLANNER_CHECKPOINT" "$LEWM_CHECKPOINT" "$REFERENCE_GUIDED_CHECKPOINT" "$ROLLOUT_CACHE_PATH"; do
    if [[ ! -f "$file" ]]; then echo "Required input is missing: $file" >&2; exit 2; fi
  done
  mkdir -p "$OUTPUT_DIR/eval"
fi

TRAIN=("${PYTHON[@]}" -m ablation.train
  data=pusht "data.dataset.name=$DATASET_NAME"
  "ablation.variant=$VARIANT" "ablation.output_dir=$OUTPUT_DIR"
  "planner_checkpoint=$PLANNER_CHECKPOINT" "lewm_checkpoint=$LEWM_CHECKPOINT"
  "rollout_cache.path=$ROLLOUT_CACHE_PATH"
  staged_training.generate_rollout_cache=false
  staged_training.distillation=true staged_training.fine_tune=true
  "staged_training.distillation_epochs=$DISTILLATION_EPOCHS"
  "staged_training.fine_tune_epochs=$FINE_TUNE_EPOCHS"
  "staged_training.distillation_lr=$DISTILLATION_LR"
  "staged_training.fine_tune_lr=$FINE_TUNE_LR"
  "loader.num_workers=$LOADER_WORKERS" "loader.batch_size=$BATCH_SIZE"
  "wandb.mode=${WANDB_MODE:-online}" "seed=$SEED" "$@")
echo "== 1/3 Train $VARIANT: existing cache; distillation then fine-tuning =="
if [[ "${DRY_RUN:-0}" == 1 ]]; then print_command "${TRAIN[@]}"; else "${TRAIN[@]}" 2>&1 | tee "$OUTPUT_DIR/train.log"; fi

if [[ "${DRY_RUN:-0}" != 1 ]]; then
  [[ -f "$ABLATION_CHECKPOINT" ]] || { echo "Training did not create $ABLATION_CHECKPOINT" >&2; exit 1; }
  printf 'seed,success_rate_percent,evaluation_time_seconds\n' > "$OUTPUT_DIR/eval/per_seed.csv"
fi
echo "== 2/3 Ten-seed evaluation: horizon=10 action_block=5 goal_offset=50 budget=100 =="
for seed in "${SEEDS[@]}"; do
  EVAL=("${PYTHON[@]}" -m ablation.evaluate solver=latent_flow
    "ablation.output_dir=$OUTPUT_DIR" "policy=$PLANNER_CHECKPOINT"
    "solver.lewm_checkpoint=$LEWM_CHECKPOINT"
    "solver.reference_guided_checkpoint=$ABLATION_CHECKPOINT"
    "eval.dataset_name=$DATASET_NAME"
    plan_config.horizon=10 plan_config.receding_horizon=10 plan_config.action_block=5
    eval.goal_offset_steps=50 eval.eval_budget=100 "eval.num_eval=$NUM_EVAL"
    eval.save_artifacts=false "seed=$seed")
  if [[ "${DRY_RUN:-0}" == 1 ]]; then print_command "${EVAL[@]}"; continue; fi
  if ! output=$("${EVAL[@]}" 2>&1); then
    printf '%s\n' "$output" | tee "$OUTPUT_DIR/eval/seed_${seed}.log"
    echo "Evaluation failed for seed=$seed" >&2; exit 1
  fi
  printf '%s\n' "$output" | tee "$OUTPUT_DIR/eval/seed_${seed}.log"
  rate=$(printf '%s\n' "$output" | sed -nE 's/.*evaluation_success_rate=([0-9.]+)%.*/\1/p' | tail -n 1)
  elapsed=$(printf '%s\n' "$output" | sed -nE 's/.*evaluation_time_seconds=([0-9.]+).*/\1/p' | tail -n 1)
  if [[ -z "$rate" || -z "$elapsed" ]]; then echo "Cannot parse seed=$seed results" >&2; exit 1; fi
  printf '%s,%s,%s\n' "$seed" "$rate" "$elapsed" >> "$OUTPUT_DIR/eval/per_seed.csv"
done
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  "${PYTHON[@]}" -m ablation.summarize "$OUTPUT_DIR/eval/per_seed.csv" --episodes-per-seed "$NUM_EVAL" | tee "$OUTPUT_DIR/eval/summary.txt"
fi

echo "== 3/3 Same-candidate analysis: ablation vs full reference-guided vs continuous LeWM =="
ANALYZE=("${PYTHON[@]}" -m ablation.analyze
  --task pusht --dataset-name "$DATASET_NAME"
  --planner-checkpoint "$PLANNER_CHECKPOINT" --lewm-checkpoint "$LEWM_CHECKPOINT"
  --reference-guided-checkpoint "$REFERENCE_GUIDED_CHECKPOINT"
  --ablation-checkpoint "$ABLATION_CHECKPOINT"
  --tasks "$ANALYSIS_TASKS" --candidates "$ANALYSIS_CANDIDATES"
  --goal-offset-steps 50 --flow-steps 16 --history-size 3 --seed "$SEED"
  --output-dir "$OUTPUT_DIR/analysis")
if [[ "${DRY_RUN:-0}" == 1 ]]; then print_command "${ANALYZE[@]}"; else "${ANALYZE[@]}" 2>&1 | tee "$OUTPUT_DIR/analysis.log"; fi
echo "Ablation results: $OUTPUT_DIR"
