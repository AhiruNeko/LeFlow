"""Shared real-rollout utilities for reference-guided LeFlow analysis."""

from __future__ import annotations

import json
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import stable_worldmodel as swm
from stable_worldmodel_compat import create_world
import torch

from latent_planner import LatentPlannerRuntime, load_lewm
from reference_guided_world_model import ReferenceGuidedWorldModel, continuous_lewm_predictions, parallel_segment_predictions
from train_reference_guided_world_model import image_transform, simulator_targets


TASK_PROFILES = {
    "pusht": {
        "dataset_keys": ["pixels", "action", "state", "proprio"],
        "world": {"env_name": "swm/PushT-v1"},
        "simulator_restore": None,
    },
    "reacher": {
        "dataset_keys": ["pixels", "action", "qpos", "qvel"],
        "world": {
            "env_name": "swm/ReacherDMControl-v0",
            "task": "qpos_match",
        },
        "simulator_restore": [
            {
                "method": "set_state",
                "args": {
                    "qpos": {"source": "qpos", "index": 0},
                    "qvel": {"source": "qvel", "index": 0},
                },
            },
            {
                "method": "set_target_qpos",
                "args": {"target_qpos": {"source": "qpos", "index": -1}},
            },
        ],
    },
}


def parse_args_common(parser):
    parser.add_argument("--planner-checkpoint", required=True)
    parser.add_argument("--reference-guided-checkpoint", required=True)
    parser.add_argument("--lewm-checkpoint", required=True, help="Exact frozen LeWM checkpoint used for every latent encoding and prediction.")
    parser.add_argument("--task", choices=sorted(TASK_PROFILES), default="pusht")
    parser.add_argument("--dataset-name", default="pusht_expert_train")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--goal-offset-steps", type=int, default=50)
    parser.add_argument("--flow-steps", type=int, default=16)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    return parser


def ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    result, sorted_values = np.empty(values.size), values[order]
    first = 0
    while first < values.size:
        last = first + 1
        while last < values.size and sorted_values[last] == sorted_values[first]:
            last += 1
        result[order[first:last]] = 0.5 * (first + last - 1)
        first = last
    return result


def pearson(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if x.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    return pearson(ranks(x), ranks(y))


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "sample_std": float(values.std(ddof=1)) if values.size > 1 else 0.0, "count": int(values.size)}


def make_dataset(dataset_name, cache_dir, task):
    profile = TASK_PROFILES[task]
    keys = profile["dataset_keys"]
    return swm.data.HDF5Dataset(
        dataset_name,
        keys_to_load=keys,
        keys_to_cache=[key for key in keys if key != "pixels"],
        cache_dir=cache_dir,
        transform=None,
    )


def choose_tasks(dataset, count, goal_offset, seed):
    try:
        episodes = np.asarray(dataset.get_col_data("episode_idx"))
    except KeyError:
        episodes = np.asarray(dataset.get_col_data("ep_idx"))
    steps = np.asarray(dataset.get_col_data("step_idx"))
    valid = []
    for episode in np.unique(episodes):
        episode_steps = steps[episodes == episode]
        max_step = int(episode_steps.max())
        valid.extend((int(episode), int(start)) for start in episode_steps if int(start) + goal_offset <= max_step)
    if len(valid) < count:
        raise ValueError(f"Only {len(valid)} clips support goal offset {goal_offset}.")
    return [valid[int(index)] for index in np.random.default_rng(seed).choice(len(valid), size=count, replace=False)]


def action_normalizer(dataset, device):
    actions = torch.as_tensor(dataset.get_col_data("action"), dtype=torch.float32)
    return actions.mean(0).to(device), actions.std(0).clamp_min(1e-6).to(device)


class RealCandidateEvaluator:
    """Generate planner candidates, execute task rollouts, and score both models."""

    def __init__(self, args):
        self.args, self.device = args, torch.device(args.device)
        self.runtime = LatentPlannerRuntime.from_checkpoint(args.planner_checkpoint, device=self.device).eval()
        # Do not silently use the LeWM path embedded in the planner checkpoint:
        # every analysis latent, baseline rollout, and local segment prediction
        # must use the explicit checkpoint supplied by the caller.
        self.runtime.lewm = load_lewm(args.lewm_checkpoint).to(self.device).eval()
        self.runtime.lewm.requires_grad_(False)
        self.runtime.requires_grad_(False)
        self.guided = ReferenceGuidedWorldModel.from_checkpoint(args.reference_guided_checkpoint, device=self.device).eval()
        self.guided.requires_grad_(False)
        self.task = str(args.task)
        self.profile = TASK_PROFILES[self.task]
        self.dataset = make_dataset(args.dataset_name, args.cache_dir, self.task)
        # flow.max_horizon is architectural capacity (20 for the H10 planner),
        # not the horizon used to train and evaluate this checkpoint.
        planner_payload = torch.load(
            args.planner_checkpoint, map_location="cpu", weights_only=False
        )
        try:
            self.horizon = int(planner_payload["config"]["planner"]["horizon"])
        except (KeyError, TypeError) as error:
            raise KeyError(
                "Planner checkpoint must provide config.planner.horizon; "
                "do not substitute flow.max_horizon."
            ) from error
        self.action_mean, self.action_std = action_normalizer(self.dataset, self.device)
        self.transform = image_transform(args.img_size)
        world_config = dict(self.profile["world"])
        world_config.update(
            num_envs=int(args.candidates),
            max_episode_steps=self.horizon * int(self.runtime.action_block),
        )
        self.world = create_world(
            world_config, image_shape=(args.img_size, args.img_size)
        )
        self.task_restore = self.profile["simulator_restore"]
        self.generator = torch.Generator(device=self.device).manual_seed(int(args.seed))

    @torch.no_grad()
    def evaluate_task(self, episode, start):
        raw = self.dataset.load_chunk([int(episode)], [int(start)], [int(start) + int(self.args.goal_offset_steps)])[0]
        pixels = torch.as_tensor(raw["pixels"])
        if pixels.ndim != 4:
            raise ValueError(f"Expected clip pixels [T,C,H,W], got {tuple(pixels.shape)}")
        z_start = self.runtime.encode_pixels(self.transform(pixels[:1]).to(self.device))[:, 0]
        z_goal = self.runtime.encode_pixels(self.transform(pixels[-1:]).to(self.device))[:, 0]
        paths = self.runtime.sample_paths(z_start, z_goal, horizon=self.horizon, num_samples=int(self.args.candidates), flow_steps=int(self.args.flow_steps), generator=self.generator)
        actions = self.runtime.decode_actions(paths)
        target = simulator_targets(
            self.runtime, self.world, self.dataset, np.asarray([episode]),
            np.asarray([start]), paths, actions,
            goal_offset=int(self.args.goal_offset_steps), image_tx=self.transform,
            action_mean=self.action_mean, action_std=self.action_std,
            device=self.device, task_restore=self.task_restore,
        )
        flat_paths, flat_actions = paths[0], actions[0]
        lewm = continuous_lewm_predictions(self.runtime.lewm, flat_paths, flat_actions, history_size=int(self.args.history_size))
        local = parallel_segment_predictions(self.runtime.lewm, flat_paths, flat_actions)
        guided = self.guided(flat_paths, flat_actions, local)["trajectory"]
        return {"target": target, "lewm": lewm, "guided": guided, "goal": z_goal.expand(int(self.args.candidates), -1)}

    def close(self):
        close = getattr(self.world, "close", None)
        if callable(close):
            close()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
