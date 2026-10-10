"""Attribute PushT inference error across planner, IDM, and LeWM.

Latent paths cannot be decoded directly to pixels.  Planner and LeWM latent
paths are therefore rendered as nearest-reference-frame projections from the
corresponding expert trajectory; titles always label those panels accordingly.
Simulator panels, in contrast, are true PushT executions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import matplotlib.pyplot as plt
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from sklearn.preprocessing import StandardScaler
from torchvision.transforms import v2 as transforms

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from latent_planner import LatentPlannerRuntime


@dataclass
class Task:
    episode: int
    start_step: int
    indices: np.ndarray
    frames: np.ndarray
    states: np.ndarray
    actions: np.ndarray


def image_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=image_size),
        ]
    )


def episode_column(dataset) -> str:
    return "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"


def build_row_lookup(dataset) -> tuple[str, dict[tuple[int, int], int]]:
    column = episode_column(dataset)
    episodes = dataset.get_col_data(column)
    steps = dataset.get_col_data("step_idx")
    return column, {
        (int(episode), int(step)): index
        for index, (episode, step) in enumerate(zip(episodes, steps))
    }


def available_task_starts(
    dataset, *, goal_offset_steps: int
) -> tuple[str, np.ndarray]:
    column, lookup = build_row_lookup(dataset)
    episodes = dataset.get_col_data(column)
    steps = dataset.get_col_data("step_idx")
    valid = [
        index
        for index, (episode, step) in enumerate(zip(episodes, steps))
        if (int(episode), int(step) + goal_offset_steps) in lookup
    ]
    return column, np.asarray(valid, dtype=np.int64)


def load_task(
    dataset,
    *,
    row_index: int,
    goal_offset_steps: int,
) -> Task:
    column, lookup = build_row_lookup(dataset)
    row = dataset.get_row_data([int(row_index)])
    episode = int(row[column][0])
    start_step = int(row["step_idx"][0])
    indices = np.asarray(
        [
            lookup[(episode, start_step + offset)]
            for offset in range(goal_offset_steps + 1)
        ],
        dtype=np.int64,
    )
    sequence = dataset.get_row_data(indices)
    return Task(
        episode=episode,
        start_step=start_step,
        indices=indices,
        frames=np.asarray(sequence["pixels"]),
        states=np.asarray(sequence["state"]),
        actions=np.asarray(sequence["action"])[:-1],
    )


def action_scaler(dataset) -> StandardScaler:
    scaler = StandardScaler()
    actions = np.asarray(dataset.get_col_data("action"))
    scaler.fit(actions[~np.isnan(actions).any(axis=1)])
    return scaler


def expert_segment_indices(
    *, horizon: int, action_block: int, goal_offset_steps: int
) -> np.ndarray:
    if horizon * action_block != goal_offset_steps:
        raise ValueError(
            "This attribution analysis requires horizon * action_block to equal "
            "goal_offset_steps so all module outputs align to one expert path."
        )
    return np.arange(horizon + 1, dtype=np.int64) * action_block


@torch.no_grad()
def encode_frames(
    runtime: LatentPlannerRuntime,
    frames: np.ndarray,
    transform,
) -> torch.Tensor:
    processed = []
    for frame in frames:
        tensor = torch.from_numpy(np.asarray(frame))
        if tensor.ndim == 3 and tensor.shape[-1] in (1, 3, 4):
            tensor = tensor.permute(2, 0, 1)
        processed.append(transform(tensor))
    model_frames = torch.stack(processed).to(runtime.device)
    return runtime.encode_pixels(model_frames)[:, -1]


def actions_to_blocks(
    actions_raw: np.ndarray, scaler: StandardScaler, *, horizon: int, action_block: int
) -> torch.Tensor:
    action_dim = actions_raw.shape[-1]
    normalised = scaler.transform(actions_raw.reshape(-1, action_dim))
    return torch.from_numpy(normalised.astype(np.float32)).reshape(
        horizon, action_block * action_dim
    )


def blocks_to_actions(
    blocks: torch.Tensor, scaler: StandardScaler, *, action_block: int
) -> np.ndarray:
    blocks_np = blocks.detach().cpu().numpy()
    raw_dim = blocks_np.shape[-1] // action_block
    raw = scaler.inverse_transform(blocks_np.reshape(-1, raw_dim))
    return raw.astype(np.float32)


@torch.no_grad()
def lewm_rollout(
    runtime: LatentPlannerRuntime,
    z_start: torch.Tensor,
    action_blocks: torch.Tensor,
    *,
    history_size: int,
) -> torch.Tensor:
    return runtime.rollout_paths(
        z_start[None],
        action_blocks[None, None].to(runtime.device),
        history_size=history_size,
    )[0, 0]


def nearest_reference_frames(
    latent_path: torch.Tensor,
    reference_latents: torch.Tensor,
    reference_frames: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray]:
    distances = torch.cdist(latent_path.float(), reference_latents.float()).pow(2)
    indices = distances.argmin(dim=1).detach().cpu().numpy()
    return [reference_frames[int(index)] for index in indices], indices


def setup_environment(task: Task, *, max_steps: int):
    world = swm.World(
        env_name="swm/PushT-v1",
        num_envs=1,
        max_episode_steps=max_steps,
        history_size=1,
        frame_skip=1,
        image_shape=(224, 224),
    )
    env = world.envs.unwrapped.envs[0].unwrapped
    env.reset()
    env._set_state(task.states[0].copy())
    env._set_goal_state(task.states[-1].copy())
    return env


def simulator_rollout(
    task: Task,
    actions_raw: np.ndarray,
    *,
    action_block: int,
) -> tuple[list[np.ndarray], bool]:
    env = setup_environment(task, max_steps=len(actions_raw) + 1)
    frames = [np.asarray(env.render()).copy()]
    success = False
    for index, action in enumerate(actions_raw):
        _, _, terminated, truncated, _ = env.step(action)
        success = bool(success or terminated)
        if (index + 1) % action_block == 0:
            frames.append(np.asarray(env.render()).copy())
        if terminated or truncated:
            break
    while len(frames) < len(actions_raw) // action_block + 1:
        frames.append(frames[-1].copy())
    return frames, success


def add_boundary(ax, color: str, linewidth: float) -> None:
    height, width = ax.images[0].get_array().shape[:2]
    rectangle = plt.Rectangle(
        (0, 0), width - 1, height - 1, fill=False, edgecolor=color, linewidth=linewidth
    )
    ax.add_patch(rectangle)


def draw_overlay(ax, frames: list[np.ndarray], title: str) -> None:
    """Overlay only rendered objects, not the white background of every frame."""
    if not frames:
        ax.set_axis_off()
        return

    base = np.asarray(frames[0])[..., :3]
    ax.imshow(base)
    count = len(frames)
    for index, frame in enumerate(frames[1:], start=1):
        rgb = np.asarray(frame)[..., :3]
        # PushT uses a nearly white canvas. Keeping that canvas transparent avoids
        # bleaching the earlier states while retaining the robot, block, and goal.
        foreground = (rgb.min(axis=-1) < 245).astype(np.float32)
        alpha = 0.58 if index < count - 1 else 0.92
        rgba = np.concatenate(
            [rgb, (foreground * alpha * 255.0)[..., None]], axis=-1
        ).astype(np.uint8)
        ax.imshow(rgba)

    add_boundary(ax, "tab:green", 3.0)
    add_boundary(ax, "tab:red", 1.6)
    ax.set_title(title, fontsize=9)
    ax.set_axis_off()

def save_comparison(
    output: Path,
    *,
    expert_frames: list[np.ndarray],
    panels: list[tuple[str, list[np.ndarray]]],
    title: str,
) -> None:
    figure, axes = plt.subplots(
        1 + len(panels), 1, figsize=(5.0, 4.4 * (1 + len(panels)))
    )
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([axes])
    draw_overlay(axes[0], expert_frames, "Expert trajectory")
    for axis, (label, frames) in zip(axes[1:], panels):
        draw_overlay(axis, frames, label)
    figure.suptitle(title, fontsize=12, y=0.995)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)


def mse_path(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).pow(2).mean().detach().cpu())


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument(
        "--planner-checkpoint",
        type=Path,
        default=Path("/root/projects/pusht/latent_planner_finetune/pusht_h10_phase2_v1/fine_tuned_latent_planner.pt"),
    )
    result.add_argument("--dataset-name", default="pusht_expert_train")
    result.add_argument("--cache-dir", type=Path, default=None)
    result.add_argument("--output-dir", type=Path, default=Path("analysis/inference_chain_finetuned"))
    result.add_argument("--num-tasks", type=int, default=50)
    result.add_argument("--render-tasks", type=int, default=12)
    result.add_argument("--planner-paths", type=int, default=4)
    result.add_argument("--horizon", type=int, default=10)
    result.add_argument("--action-block", type=int, default=5)
    result.add_argument("--goal-offset-steps", type=int, default=50)
    result.add_argument("--flow-steps", type=int, default=16)
    result.add_argument("--history-size", type=int, default=3)
    result.add_argument("--latent-success-threshold", type=float, default=0.10)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--device", default="cuda")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.num_tasks < 1 or args.planner_paths < 1:
        raise ValueError("num-tasks and planner-paths must be positive")
    segment_steps = expert_segment_indices(
        horizon=args.horizon,
        action_block=args.action_block,
        goal_offset_steps=args.goal_offset_steps,
    )

    requested = torch.device(args.device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        requested = torch.device("cpu")
    runtime = LatentPlannerRuntime.from_checkpoint(
        args.planner_checkpoint, device=requested
    ).eval()
    runtime.requires_grad_(False)
    if runtime.action_block != args.action_block:
        raise ValueError(
            f"Checkpoint action_block={runtime.action_block}, requested {args.action_block}"
        )

    cache_dir = args.cache_dir or Path(swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        args.dataset_name,
        keys_to_cache=["action", "state"],
        cache_dir=cache_dir,
    )
    _, valid_indices = available_task_starts(
        dataset, goal_offset_steps=args.goal_offset_steps
    )
    if len(valid_indices) < args.num_tasks:
        raise ValueError("Dataset does not contain enough valid analysis tasks")

    rng = np.random.default_rng(args.seed)
    chosen_rows = np.sort(
        rng.choice(valid_indices, size=args.num_tasks, replace=False)
    )
    output_dir = args.output_dir / time.strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=False)
    transform = image_transform(224)
    scaler = action_scaler(dataset)
    torch_generator = torch.Generator(device=requested).manual_seed(args.seed)

    summary: dict[str, list[float]] = {
        "planner_interior_mse": [],
        "planner_first_interior_mse": [],
        "idm_expert_sim_success": [],
        "lewm_expert_action_terminal_mse": [],
        "lewm_expert_action_latent_success": [],
        "expert_path_idm_lewm_terminal_mse": [],
        "expert_path_idm_lewm_latent_success": [],
        "planner_idm_sim_success": [],
    }
    records: list[dict[str, Any]] = []

    for task_number, row_index in enumerate(chosen_rows):
        task = load_task(
            dataset, row_index=int(row_index), goal_offset_steps=args.goal_offset_steps
        )
        expert_frames = task.frames[segment_steps]
        expert_latents = encode_frames(runtime, expert_frames, transform)
        z_start = expert_latents[0:1]
        z_goal = expert_latents[-1:]
        expert_actions_raw = task.actions[: args.horizon * args.action_block]
        expert_blocks = actions_to_blocks(
            expert_actions_raw,
            scaler,
            horizon=args.horizon,
            action_block=args.action_block,
        ).to(requested)

        with torch.no_grad():
            planner_paths = runtime.sample_paths(
                z_start,
                z_goal,
                horizon=args.horizon,
                num_samples=args.planner_paths,
                flow_steps=args.flow_steps,
                generator=torch_generator,
            )[0]
            expert_idm_blocks = runtime.decode_actions(
                expert_latents[None, None]
            )[0, 0]
            lewm_expert = lewm_rollout(
                runtime, z_start[0], expert_blocks, history_size=args.history_size
            )
            lewm_expert_idm = lewm_rollout(
                runtime,
                z_start[0],
                expert_idm_blocks,
                history_size=args.history_size,
            )

        planner_errors = (
            (planner_paths[:, 1:-1] - expert_latents[None, 1:-1])
            .pow(2)
            .mean(dim=(-1, -2))
            .detach()
            .cpu()
            .numpy()
        )
        idm_actions_raw = blocks_to_actions(
            expert_idm_blocks, scaler, action_block=args.action_block
        )
        planner_idm_blocks = runtime.decode_actions(planner_paths[0:1, None])[0, 0]
        planner_actions_raw = blocks_to_actions(
            planner_idm_blocks, scaler, action_block=args.action_block
        )

        idm_frames, idm_success = simulator_rollout(
            task, idm_actions_raw, action_block=args.action_block
        )
        planner_frames, planner_success = simulator_rollout(
            task, planner_actions_raw, action_block=args.action_block
        )
        lewm_expert_frames, lewm_expert_nearest = nearest_reference_frames(
            lewm_expert, expert_latents, expert_frames
        )
        lewm_expert_idm_frames, lewm_expert_idm_nearest = nearest_reference_frames(
            lewm_expert_idm, expert_latents, expert_frames
        )

        lewm_expert_terminal = mse_path(lewm_expert[-1], expert_latents[-1])
        lewm_expert_idm_terminal = mse_path(
            lewm_expert_idm[-1], expert_latents[-1]
        )
        summary["planner_interior_mse"].append(float(planner_errors.mean()))
        summary["planner_first_interior_mse"].append(float(planner_errors[0]))
        summary["idm_expert_sim_success"].append(float(idm_success))
        summary["lewm_expert_action_terminal_mse"].append(lewm_expert_terminal)
        summary["lewm_expert_action_latent_success"].append(
            float(lewm_expert_terminal <= args.latent_success_threshold)
        )
        summary["expert_path_idm_lewm_terminal_mse"].append(
            lewm_expert_idm_terminal
        )
        summary["expert_path_idm_lewm_latent_success"].append(
            float(lewm_expert_idm_terminal <= args.latent_success_threshold)
        )
        summary["planner_idm_sim_success"].append(float(planner_success))

        record = {
            "task": task_number,
            "episode": task.episode,
            "start_step": task.start_step,
            "planner_path_interior_mse": planner_errors.tolist(),
            "idm_expert_sim_success": bool(idm_success),
            "lewm_expert_action_terminal_mse": lewm_expert_terminal,
            "lewm_expert_action_latent_success": bool(
                lewm_expert_terminal <= args.latent_success_threshold
            ),
            "expert_path_idm_lewm_terminal_mse": lewm_expert_idm_terminal,
            "expert_path_idm_lewm_latent_success": bool(
                lewm_expert_idm_terminal <= args.latent_success_threshold
            ),
            "planner_first_idm_sim_success": bool(planner_success),
            "lewm_expert_nearest_frames": lewm_expert_nearest.tolist(),
            "expert_idm_lewm_nearest_frames": lewm_expert_idm_nearest.tolist(),
        }
        records.append(record)

        if task_number < args.render_tasks:
            method_dirs = {
                "planner_paths": output_dir / "planner_paths",
                "idm_expert_to_simulator": output_dir / "idm_expert_to_simulator",
                "lewm_expert_actions": output_dir / "lewm_expert_actions",
                "expert_path_idm_to_lewm": output_dir / "expert_path_idm_to_lewm",
                "planner_idm_to_simulator": output_dir / "planner_idm_to_simulator",
            }
            for method_dir in method_dirs.values():
                method_dir.mkdir(parents=True, exist_ok=True)

            planner_panels = []
            for path_index, path in enumerate(planner_paths):
                frames, nearest = nearest_reference_frames(
                    path, expert_latents, expert_frames
                )
                planner_panels.append(
                    (
                        "Planner path "
                        f"{path_index + 1}: interior MSE={planner_errors[path_index]:.4g}; "
                        "nearest expert-frame projection",
                        frames,
                    )
                )
                record.setdefault("planner_nearest_frames", []).append(
                    nearest.tolist()
                )

            save_comparison(
                method_dirs["planner_paths"] / f"task_{task_number:03d}.png",
                expert_frames=list(expert_frames),
                panels=planner_panels,
                title=(
                    f"Task {task_number}: raw planner latent paths "
                    "(green=start, red=goal)"
                ),
            )
            save_comparison(
                method_dirs["idm_expert_to_simulator"] / f"task_{task_number:03d}.png",
                expert_frames=list(expert_frames),
                panels=[
                    (f"IDM(expert path) -> simulator; success={idm_success}", idm_frames),
                ],
                title=f"Task {task_number}: expert path -> IDM -> simulator",
            )
            save_comparison(
                method_dirs["lewm_expert_actions"] / f"task_{task_number:03d}.png",
                expert_frames=list(expert_frames),
                panels=[
                    (
                        "LeWM(expert actions); terminal latent "
                        f"MSE={lewm_expert_terminal:.4g}; nearest-frame projection",
                        lewm_expert_frames,
                    ),
                ],
                title=f"Task {task_number}: expert actions -> LeWM",
            )
            save_comparison(
                method_dirs["expert_path_idm_to_lewm"] / f"task_{task_number:03d}.png",
                expert_frames=list(expert_frames),
                panels=[
                    (
                        "IDM(expert path) -> LeWM; terminal latent "
                        f"MSE={lewm_expert_idm_terminal:.4g}; nearest-frame projection",
                        lewm_expert_idm_frames,
                    ),
                ],
                title=f"Task {task_number}: expert path -> IDM -> LeWM",
            )
            save_comparison(
                method_dirs["planner_idm_to_simulator"] / f"task_{task_number:03d}.png",
                expert_frames=list(expert_frames),
                panels=[
                    (
                        "Planner path 1 -> IDM -> simulator; "
                        f"success={planner_success}",
                        planner_frames,
                    ),
                ],
                title=f"Task {task_number}: planner path -> IDM -> simulator",
            )
        print(
            f"task={task_number} episode={task.episode} start={task.start_step} "
            f"planner_mse={planner_errors.mean():.4g} "
            f"idm_success={int(idm_success)} "
            f"lewm_expert_terminal={lewm_expert_terminal:.4g} "
            f"expert_idm_lewm_terminal={lewm_expert_idm_terminal:.4g} "
            f"planner_idm_success={int(planner_success)}",
            flush=True,
        )

    aggregate = {
        key: float(np.mean(values)) if values else float("nan")
        for key, values in summary.items()
    }
    payload = {
        "checkpoint": str(args.planner_checkpoint),
        "dataset": args.dataset_name,
        "horizon": args.horizon,
        "action_block": args.action_block,
        "goal_offset_steps": args.goal_offset_steps,
        "flow_steps": args.flow_steps,
        "num_tasks": args.num_tasks,
        "planner_paths": args.planner_paths,
        "latent_success_threshold": args.latent_success_threshold,
        "aggregate": aggregate,
        "records": records,
        "render_note": (
            "Simulator panels are true PushT rollouts. Planner and LeWM panels "
            "are nearest-reference-frame latent projections because this model "
            "has no latent-to-image decoder."
        ),
    }
    with (output_dir / "results.json").open("w") as file:
        json.dump(payload, file, indent=2)
    with (output_dir / "summary.txt").open("w") as file:
        file.write(json.dumps(payload["aggregate"], indent=2))
        file.write("\n\n")
        file.write(payload["render_note"])
        file.write("\n")
    print(json.dumps(aggregate, indent=2))
    print(f"Saved analysis to {output_dir}")


if __name__ == "__main__":
    main()
