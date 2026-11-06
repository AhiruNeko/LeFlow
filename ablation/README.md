# Reference-guided ablations

All new source, configurations, scripts, checkpoints, logs and analysis outputs
live below this directory. Existing project files are not edited.

## Variants

- `no_local`: remove both independent LeWM prediction and residual FiLM branches.
  Keep the complete planner-path memory, global start/goal AdaLN, action tokens,
  all four cross/self-attention blocks and the start-anchored trajectory head.
- `continuous_condition`: retain the complete reference-guided architecture and
  replace independent segment LeWM predictions with the frozen LeWM continuous
  rollout. The prediction and residual FiLM branches both receive this continuous
  condition; the residual is computed against planner path points as before.
  Distillation still supervises against that same continuous rollout, so this
  variant can learn an identity shortcut. Fine-tuning still uses simulator cache.
- `endpoints_actions`: additionally remove all interior planner states. The same
  cross-attention blocks read only two endpoint tokens, using their original
  positions (0 and H). Global conditioning, temporal attention, FFN and head are
  unchanged. This is an endpoint-conditioned model, not an action-only model.

`no_local` and `endpoints_actions` remove local modules from their checkpoints and ignore
local predictions; `endpoints_actions` also ignores the interior path. The continuous
variant keeps all original modules, adds a checkpoint marker, and uses the same
continuous LeWM inference during training, evaluation, and analysis. Planner/IDM
still generate candidates. The comparison analysis computes independent local
LeWM predictions only for the full-model baseline.

## PushT pipelines

Run from the repository (the scripts also locate it automatically):

```bash
export STABLEWM_HOME=/root/autodl-tmp/pusht
export PLANNER_CHECKPOINT=/path/to/latent_planner_h10.pt
export LEWM_CHECKPOINT=/path/to/lewm_object.ckpt
export REFERENCE_GUIDED_CHECKPOINT=/path/to/full_reference_guided_world_model_best.pt
export ROLLOUT_CACHE_PATH=/path/to/pusht_h10_staged_rollout_cache.h5

bash ablation/shell/train_pusht_no_local.sh
bash ablation/shell/train_pusht_endpoints_actions.sh
bash ablation/shell/train_pusht_continuous_condition.sh
```

Each pipeline performs:

1. Reuse the specified cache if present; otherwise generate it from the fine-tuning
   split before training. Distillation uses the original 90% source partition;
   fine-tuning uses 90% of the cached 10% partition, following the existing trainer. Defaults: distillation 10 epochs at
   1e-4; fine-tuning 25 epochs at 3e-5. Full-path equal-weight MSE is unchanged.
   The original source HDF5 is still required for distillation. Training uses the
   explicitly supplied LeWM, including when the planner embeds another path.
2. Evaluate the fine-tune best checkpoint on seeds 42-51, 50 episodes each. Horizon
   and receding horizon 10, action block 5, goal offset 50, evaluation budget 100,
   64 planner candidates, 16 flow steps. Save per-seed CSV/logs and mean/sample std.
3. Compare that checkpoint with the specified full reference-guided checkpoint
   and continuous LeWM. All three share the same 10 tasks, 64 candidates, actions
   and real simulator rollouts. Save full/endpoint MSE, goal-error Pearson/Spearman
   globally and per task, candidate selection regret/top-1 match, CSV/JSON and PNG.
   Clip/goal selection follows the existing analysis utility.

Outputs default to `ablation/outputs/pusht/<variant>/<timestamp>/`. The trained
checkpoint is `train/ablation_<variant>_best.pt`, selected by fine-tune validation
full-trajectory MSE as in the original trainer. Fine-tuning starts from the last
in-memory distilled epoch, also unchanged. Metadata stores `config.ablation.variant`;
use the ablation evaluation entry, not the full-model checkpoint loader.

Optional environment overrides: `CONDA_ENV` (default `lewm`), `PYTHON_BIN` (bypass
conda), `DATASET_NAME`, `SEED`, `NUM_EVAL`, `EVAL_SEEDS` (ten space-separated seeds),
`DISTILLATION_EPOCHS`, `FINE_TUNE_EPOCHS`, `DISTILLATION_LR`, `FINE_TUNE_LR`,
`LOADER_WORKERS`, `BATCH_SIZE`, `ANALYSIS_TASKS`, `ANALYSIS_CANDIDATES`, `WANDB_MODE`,
`RUN_TAG`, `OUTPUT_DIR` (must stay under `ablation/`). Extra script arguments are
training Hydra overrides only. Do not disable training stages in this full pipeline.

Print all commands without creating outputs or starting work:

```bash
DRY_RUN=1 bash ablation/shell/train_pusht_no_local.sh
DRY_RUN=1 bash ablation/shell/train_pusht_endpoints_actions.sh
DRY_RUN=1 bash ablation/shell/train_pusht_continuous_condition.sh
```

The adapters substitute model/step/solver dependencies only within their own
process-scoped contexts and restore them on exit. The original trainer executes
all splits, optimizer steps, validation and checkpoint selection. The original
evaluator executes all task sampling, candidate ranking and policy execution.
`stable-worldmodel`/MuJoCo fixes are not altered.

## Checks

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover -s ablation/tests -v
python -B -m ablation.train --cfg job --resolve
python -B -m ablation.evaluate solver=latent_flow --cfg job --resolve
```

Behavior tests include removed-input invariance, shared-backbone equivalence,
gradients, checkpoint round trips and wrong-variant rejection, both actual staged
training loops on tiny synthetic data, three-way metric output, and a real local
PushT planner-checkpoint loading smoke test when its files are available. They
clean up their temporary fixtures. Full benchmark runs require the source PushT
HDF5 and model checkpoints. The cache may be supplied or generated on the
execution machine.
