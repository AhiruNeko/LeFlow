import os

os.environ["MUJOCO_GL"] = "egl"

import json
import time
import tempfile
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm

class TerminalHorizonWorldModelPolicy(swm.policy.WorldModelPolicy):
    """World-model MPC policy with a short final replan.

    Planning uses the configured (H, K) pair until at most one execution
    window remains. The final solve then uses H = K = remaining action blocks,
    so its scored endpoint is exactly the endpoint that will be executed.
    """

    def __init__(self, *args, eval_budget: int, **kwargs):
        super().__init__(*args, **kwargs)
        self._base_config = self.cfg
        self._eval_budget = int(eval_budget)
        self._remaining_env_steps = self._eval_budget
        self._terminal_plan_announced = False

    def set_env(self, env):
        super().set_env(env)
        self._remaining_env_steps = self._eval_budget

    def _terminal_config(self):
        action_block = self._base_config.action_block
        remaining_blocks = max(
            1, int(np.ceil(self._remaining_env_steps / action_block))
        )
        terminal_horizon = min(self._base_config.horizon, remaining_blocks)
        return swm.PlanConfig(
            horizon=terminal_horizon,
            receding_horizon=terminal_horizon,
            history_len=self._base_config.history_len,
            action_block=action_block,
            warm_start=self._base_config.warm_start,
        )

    def _maybe_use_terminal_config(self):
        execution_window_steps = (
            self._base_config.receding_horizon * self._base_config.action_block
        )
        if self._remaining_env_steps > execution_window_steps:
            return self._base_config

        config = self._terminal_config()
        if not self._terminal_plan_announced:
            print(
                "terminal_replan: "
                f"remaining_steps={self._remaining_env_steps}, "
                f"using horizon=receding_horizon={config.horizon}",
                flush=True,
            )
            self._terminal_plan_announced = True
        self.solver.configure(
            action_space=self.env.action_space,
            n_envs=getattr(self.env, "num_envs", 1),
            config=config,
        )
        return config

    def get_action(self, info_dict: dict, **kwargs):
        assert hasattr(self, "env"), "Environment not set for the policy"
        assert "pixels" in info_dict, "'pixels' must be provided in info_dict"
        assert "goal" in info_dict, "'goal' must be provided in info_dict"

        info_dict = self._prepare_info(info_dict)

        if len(self._action_buffer) == 0:
            active_config = self._maybe_use_terminal_config()
            outputs = self.solver(info_dict, init_action=self._next_init)
            actions = outputs["actions"]

            keep_horizon = active_config.receding_horizon
            plan = actions[:, :keep_horizon]
            rest = actions[:, keep_horizon:]
            self._next_init = rest if active_config.warm_start else None

            plan = plan.reshape(
                self.env.num_envs,
                keep_horizon * active_config.action_block,
                -1,
            )
            self._action_buffer.extend(plan.transpose(0, 1))

        action = self._action_buffer.popleft()
        action = action.reshape(*self.env.action_space.shape)
        action = action.numpy()
        if "action" in self.process:
            action = self.process["action"].inverse_transform(action)

        self._remaining_env_steps = max(0, self._remaining_env_steps - 1)
        return action


def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    return dataset


def load_heldout_episodes(cfg: DictConfig) -> set[int] | None:
    split_cfg = cfg.get("episode_split")
    if not split_cfg or not split_cfg.get("enabled", False):
        return None
    if not split_cfg.get("split_file"):
        raise ValueError("episode_split.enabled=true requires episode_split.split_file for eval.")

    split_file = Path(split_cfg.split_file)
    with split_file.open("r") as f:
        payload = json.load(f)

    mode = split_cfg.get("mode", "eval")
    key = "eval_episodes" if mode == "eval" else "train_episodes"
    episodes = {int(x) for x in payload[key]}
    if not episodes:
        raise ValueError(f"No {key} found in episode split file {split_file}.")
    print(
        f"episode_split enabled mode={mode} split_file={split_file} "
        f"episodes={len(episodes)}",
        flush=True,
    )
    return episodes


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    # create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))

    # create the transform
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)
    heldout_episodes = load_heldout_episodes(cfg)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    # -- run evaluation
    policy = cfg.get("policy", "random")

    if policy != "random":
        config = swm.PlanConfig(**cfg.plan_config)
        solver_target = cfg.solver.get("_target_", "")
        if solver_target in {
            "latent_planner.LearnedLatentPathSolver",
            "action_flow_planner.LearnedActionFlowSolver",
        }:
            solver = hydra.utils.instantiate(cfg.solver)
        else:
            model = swm.policy.AutoCostModel(cfg.policy)
            model = model.to("cuda")
            model = model.eval()
            model.requires_grad_(False)
            model.interpolate_pos_encoding = True
            solver = hydra.utils.instantiate(cfg.solver, model=model)
        policy = TerminalHorizonWorldModelPolicy(
            solver=solver,
            config=config,
            process=process,
            transform=transform,
            eval_budget=cfg.eval.eval_budget,
        )

    else:
        policy = swm.policy.RandomPolicy()

    results_path = (
        Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
        if cfg.policy != "random"
        else Path(__file__).parent
    )

    # sample the episodes and the starting indices
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    # Map each dataset row’s episode_idx to its max_start_idx
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    # remove all the lines of dataset for which dataset['step_idx'] > max_start_per_row
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    if heldout_episodes is not None:
        episode_col = dataset.get_col_data(col_name)
        valid_mask &= np.isin(episode_col, list(heldout_episodes))
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )

    # sort increasingly to avoid issues with HDF5Dataset indexing
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    print(random_episode_indices)

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)

    start_time = time.time()
    save_artifacts = bool(cfg.eval.get("save_artifacts", True))
    if save_artifacts:
        metrics = world.evaluate_from_dataset(
            dataset,
            start_steps=eval_start_idx.tolist(),
            goal_offset_steps=cfg.eval.goal_offset_steps,
            eval_budget=cfg.eval.eval_budget,
            episodes_idx=eval_episodes.tolist(),
            callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
            video_path=results_path,
        )
    else:
        # Keep evaluation outputs out of the checkpoint directory in repeated
        # multi-seed runs. The temporary directory is discarded immediately.
        with tempfile.TemporaryDirectory(prefix="leflow_eval_") as temporary_video_dir:
            metrics = world.evaluate_from_dataset(
                dataset,
                start_steps=eval_start_idx.tolist(),
                goal_offset_steps=cfg.eval.goal_offset_steps,
                eval_budget=cfg.eval.eval_budget,
                episodes_idx=eval_episodes.tolist(),
                callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
                video_path=Path(temporary_video_dir),
            )
    end_time = time.time()

    print(metrics)
    if "success_rate" in metrics:
        success_rate = float(metrics["success_rate"])
        if success_rate <= 1.0:
            success_rate *= 100.0
        print(f"evaluation_success_rate={success_rate:.2f}%", flush=True)

    if save_artifacts:
        results_path = results_path / cfg.output.filename
        results_path.parent.mkdir(parents=True, exist_ok=True)

        with results_path.open("a") as f:
            f.write("\n")  # separate from previous runs

            f.write("==== CONFIG ====\n")
            f.write(OmegaConf.to_yaml(cfg))
            f.write("\n")

            f.write("==== RESULTS ====\n")
            f.write(f"metrics: {metrics}\n")
            f.write(f"evaluation_time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    run()
