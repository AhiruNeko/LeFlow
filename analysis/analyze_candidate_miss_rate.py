#!/usr/bin/env python3
"""Measure missed successful candidates for LeWM and L2GWM; stdout only."""
from __future__ import annotations

import argparse
from copy import copy
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis.reference_guided_real_rollout_common import RealCandidateEvaluator, TASK_PROFILES
from reference_guided_world_model import continuous_lewm_predictions, parallel_segment_predictions
from train_reference_guided_world_model import (
    _disable_legacy_envpool_autoreset,
    _keep_reacher_rollout_continuous,
    init_world_from_dataset,
    set_random_seed,
)


def choose_clips(dataset, count, goal_offset, seed):
    """Sample distinct clips from valid episode lengths, without index-column padding."""
    lengths = np.asarray(dataset.lengths, dtype=np.int64)
    counts = np.maximum(lengths - goal_offset, 0)
    cumulative = np.cumsum(counts)
    total = int(cumulative[-1]) if len(cumulative) else 0
    if total < count:
        raise ValueError(f"Only {total} distinct clips support goal offset {goal_offset}")
    chosen = np.random.default_rng(seed).choice(total, size=count, replace=False)
    episodes = np.searchsorted(cumulative, chosen, side="right")
    previous = np.concatenate(([0], cumulative[:-1]))
    return [(int(ep), int(index - previous[ep])) for index, ep in zip(chosen, episodes)]


def terminal_goal_mse(trajectory, goal, candidates):
    if trajectory.ndim != 3 or goal.ndim != 2:
        raise ValueError("Expected trajectories [candidates,horizon,latent] and goals [candidates,latent]")
    if trajectory.shape[0] != candidates or trajectory.shape[2] != goal.shape[1]:
        raise ValueError("Candidate count or latent dimension does not match")
    scores = (trajectory[:, -1] - goal).square().mean(dim=-1)
    if not torch.isfinite(scores).all():
        raise ValueError("Non-finite terminal-to-goal MSE; check dataset and prediction")
    return scores.detach().cpu().numpy()


@torch.no_grad()
def prepare_candidates(evaluator, episode, start, args):
    """Generate one shared candidate pool and both models' selected indices."""
    raw = evaluator.dataset.load_chunk(
        [int(episode)], [int(start)], [int(start) + int(args.goal_offset_steps)]
    )[0]
    pixels = torch.as_tensor(raw["pixels"])
    if pixels.ndim != 4:
        raise ValueError(f"Expected clip pixels [T,C,H,W], got {tuple(pixels.shape)}")
    z_start = evaluator.runtime.encode_pixels(
        evaluator.transform(pixels[:1]).to(evaluator.device)
    )[:, 0]
    z_goal = evaluator.runtime.encode_pixels(
        evaluator.transform(pixels[-1:]).to(evaluator.device)
    )[:, 0]
    paths = evaluator.runtime.sample_paths(
        z_start, z_goal, horizon=evaluator.horizon,
        num_samples=args.candidates, flow_steps=args.flow_steps,
        generator=evaluator.generator,
    )
    actions = evaluator.runtime.decode_actions(paths)
    flat_paths, flat_actions = paths[0], actions[0]
    lewm = continuous_lewm_predictions(
        evaluator.runtime.lewm, flat_paths, flat_actions,
        history_size=args.history_size,
    )
    local = parallel_segment_predictions(
        evaluator.runtime.lewm, flat_paths, flat_actions
    )
    guided = evaluator.guided(flat_paths, flat_actions, local)["trajectory"]
    goals = z_goal.expand(args.candidates, -1)
    selected = {
        "lewm": int(np.argmin(terminal_goal_mse(lewm, goals, args.candidates))),
        "l2gwm": int(np.argmin(terminal_goal_mse(guided, goals, args.candidates))),
    }

    action_block = int(evaluator.runtime.action_block)
    action_dim = int(actions.shape[-1])
    if action_dim % action_block:
        raise ValueError("IDM action dimension must be divisible by action_block")
    env_actions = actions[0].reshape(
        args.candidates, evaluator.horizon, action_block, action_dim // action_block
    )
    env_actions = (
        env_actions * evaluator.action_std + evaluator.action_mean
    ).detach().cpu().numpy()
    return selected, env_actions


def candidate_succeeds(evaluator, episode, start, env_actions, goal_offset):
    """Re-run one candidate from the same restored state, checking every step."""
    world = evaluator.world
    init_world_from_dataset(
        world, evaluator.rollout_dataset,
        np.asarray([episode]), np.asarray([start]),
        goal_offset, evaluator.task_restore,
    )
    for segment_actions in env_actions:
        for action in segment_actions:
            _keep_reacher_rollout_continuous(world, evaluator.task_restore)
            (
                world.states, world.rewards, world.terminateds,
                world.truncateds, world.infos,
            ) = world.envs.step(action[None])
            _disable_legacy_envpool_autoreset(world, 1)
            # World.evaluate also counts terminated at any step as success.
            if np.asarray(world.terminateds, dtype=bool).any():
                return True
    return False


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(TASK_PROFILES), required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--planner-checkpoint", required=True)
    parser.add_argument("--lewm-checkpoint", required=True)
    parser.add_argument("--reference-guided-checkpoint", required=True)
    parser.add_argument("--expected-horizon", type=int, default=None)
    parser.add_argument("--goal-offset-steps", type=int, required=True)
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--candidates", type=int, default=64)
    parser.add_argument("--flow-steps", type=int, default=16)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.goal_offset_steps < 1 or args.tasks < 1 or args.candidates < 1:
        parser.error("--goal-offset-steps, --tasks, and --candidates must be positive")
    args.cache_dir = None
    return args


def main():
    args = parse_args()
    set_random_seed(args.seed)
    # Candidate generation uses args.candidates; only the simulator has one env
    # because successful candidates are checked lazily and independently.
    simulator_args = copy(args)
    simulator_args.candidates = 1
    evaluator = RealCandidateEvaluator(simulator_args)
    try:
        if args.expected_horizon is not None and evaluator.horizon != args.expected_horizon:
            raise ValueError(
                f"Planner checkpoint horizon={evaluator.horizon}, expected={args.expected_horizon}"
            )
        clips = choose_clips(
            evaluator.dataset, args.tasks, args.goal_offset_steps, args.seed
        )
        print(
            f"task={args.task} horizon={evaluator.horizon} "
            f"goal_offset_steps={args.goal_offset_steps} "
            f"tasks={args.tasks} candidates_per_task={args.candidates}",
            flush=True,
        )
        feasible_tasks = 0
        missed = {"lewm": 0, "l2gwm": 0}
        for index, (episode, start) in enumerate(clips, 1):
            selected, env_actions = prepare_candidates(evaluator, episode, start, args)
            tested = {}
            # Test both selected candidates first; shared selections are simulated once.
            for candidate in dict.fromkeys(selected.values()):
                tested[candidate] = candidate_succeeds(
                    evaluator, episode, start, env_actions[candidate],
                    args.goal_offset_steps,
                )
            # Only if both selections fail, search for any successful candidate.
            if not any(tested.values()):
                for candidate in range(args.candidates):
                    if candidate in tested:
                        continue
                    tested[candidate] = candidate_succeeds(
                        evaluator, episode, start, env_actions[candidate],
                        args.goal_offset_steps,
                    )
                    if tested[candidate]:
                        break
            feasible = any(tested.values())
            feasible_tasks += int(feasible)
            selected_success = {
                name: bool(tested[candidate]) for name, candidate in selected.items()
            }
            if feasible:
                for name, success in selected_success.items():
                    missed[name] += int(not success)
            print(
                f"task={index:02d}/{args.tasks} episode={episode} start={start} "
                f"oracle_success={int(feasible)} "
                f"lewm_selected_success={int(selected_success['lewm'])} "
                f"l2gwm_selected_success={int(selected_success['l2gwm'])} "
                f"candidates_simulated={len(tested)}",
                flush=True,
            )
        print(
            f"total_tasks={args.tasks} "
            f"feasible_tasks={feasible_tasks} "
            f"lewm_selected_success_tasks={feasible_tasks - missed['lewm']} "
            f"l2gwm_selected_success_tasks={feasible_tasks - missed['l2gwm']} "
            f"lewm_missed={missed['lewm']} "
            f"l2gwm_missed={missed['l2gwm']}",
            flush=True,
        )
        for name in ("lewm", "l2gwm"):
            rate = missed[name] / feasible_tasks if feasible_tasks else float("nan")
            print(f"{name}_miss_rate={rate:.6f}", flush=True)
    finally:
        evaluator.close()


if __name__ == "__main__":
    main()
