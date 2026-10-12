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


def img_transform(cfg):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    return np.array([np.max(step_idx[episode_idx == ep_id]) + 1 for ep_id in episodes])


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    return swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )


def load_heldout_episodes(cfg: DictConfig) -> set[int] | None:
    split_cfg = cfg.get("episode_split")
    if not split_cfg or not split_cfg.get("enabled", False):
        return None
    if not split_cfg.get("split_file"):
        raise ValueError("episode_split.enabled=true requires episode_split.split_file for eval.")

    with Path(split_cfg.split_file).open("r") as f:
        payload = json.load(f)
    key = "eval_episodes" if split_cfg.get("mode", "eval") == "eval" else "train_episodes"
    episodes = {int(x) for x in payload[key]}
    if not episodes:
        raise ValueError(f"No {key} found in episode split file {split_cfg.split_file}.")
    print(
        f"episode_split enabled mode={split_cfg.get('mode', 'eval')} "
        f"split_file={split_cfg.split_file} episodes={len(episodes)}",
        flush=True,
    )
    return episodes


@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """Evaluate a joint generator while original LeWM ranks its action candidates."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(224, 224))
    transform = {"pixels": img_transform(cfg), "goal": img_transform(cfg)}

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)
    heldout_episodes = load_heldout_episodes(cfg)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col == "pixels":
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        processor.fit(col_data[~np.isnan(col_data).any(axis=1)])
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = processor

    if cfg.get("policy", "random") == "random":
        raise ValueError("This evaluator requires a joint path--action policy checkpoint.")

    config = swm.PlanConfig(**cfg.plan_config)
    solver = hydra.utils.instantiate(cfg.solver)
    if getattr(solver.model.flow, "action_dim", None) is None:
        raise ValueError("This evaluator requires a joint path--action checkpoint.")

    policy = swm.policy.WorldModelPolicy(
        solver=solver, config=config, process=process, transform=transform
    )
    generation_model = "fast-lewm" if getattr(solver.model.lewm, "is_fast_lewm", False) else "lewm"
    print(
        "latent planner mode=joint_path_action "
        f"action_dim={solver.model.flow.action_dim} "
        f"generation_world_model={generation_model} ranker_world_model=lewm",
        flush=True,
    )

    results_path = Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    if heldout_episodes is not None:
        valid_mask &= np.isin(dataset.get_col_data(col_name), list(heldout_episodes))
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    rng = np.random.default_rng(cfg.seed)
    sampled = rng.choice(len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False)
    random_episode_indices = np.sort(valid_indices[sampled])
    print(random_episode_indices)

    rows = dataset.get_row_data(random_episode_indices)
    eval_episodes = rows[col_name]
    eval_start_idx = rows["step_idx"]
    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)
    save_artifacts = bool(cfg.eval.get("save_artifacts", True))
    temporary_video_dir = None
    if save_artifacts:
        video_path = results_path
    else:
        temporary_video_dir = tempfile.TemporaryDirectory(prefix="leflow_eval_")
        video_path = Path(temporary_video_dir.name)

    start_time = time.time()
    metrics = world.evaluate_from_dataset(
        dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset_steps=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video_path=video_path,
    )
    if temporary_video_dir is not None:
        temporary_video_dir.cleanup()
    end_time = time.time()
    print(metrics)

    if save_artifacts:
        results_path = results_path / cfg.output.filename
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with results_path.open("a") as f:
            f.write("\n==== CONFIG ====\n")
            f.write(OmegaConf.to_yaml(cfg))
            f.write("\n==== RESULTS ====\n")
            f.write(f"metrics: {metrics}\n")
            f.write(f"evaluation_time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    run()
