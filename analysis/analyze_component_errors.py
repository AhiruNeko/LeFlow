"""Decompose current-horizon errors of planner, LeWM, and inverse dynamics.

The three measurements share one demonstration task:
  * planner: sampled latent paths versus the aligned expert latent trajectory;
  * LeWM: rollout of *expert actions* versus that expert latent trajectory;
  * inverse dynamics: expert latent transitions versus normalized expert actions.

This isolates model components. In particular, planner sampling is intentionally
unconditioned on FIFO experience, and LeWM receives ground-truth actions.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import matplotlib.pyplot as plt
import numpy as np
import torch
import stable_worldmodel as swm

from latent_planner import LatentPlannerRuntime
from utils import get_img_preprocessor


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-name", default="pusht_expert_train")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--goal-offset-steps", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--planner-samples", type=int, default=64)
    parser.add_argument("--flow-steps", type=int, default=16)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--noise-seed", type=int, default=20260928)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("analysis/component_errors"))
    return parser.parse_args()


@torch.no_grad()
def image_latent(runtime, image, preprocessor, device):
    pixels = torch.as_tensor(image)
    if pixels.ndim == 3 and pixels.shape[-1] in (1, 3):
        pixels = pixels.permute(2, 0, 1)
    if pixels.ndim != 3:
        raise ValueError(f"Expected one image, got {tuple(pixels.shape)}")
    pixels = preprocessor({"pixels": pixels[None]})["pixels"][None].to(device)
    return runtime.lewm.encode({"pixels": pixels})["emb"][:, -1]


def finite_action_statistics(dataset):
    values = torch.as_tensor(np.asarray(dataset.get_col_data("action")), dtype=torch.float32)
    values = values[torch.isfinite(values).all(dim=1)]
    if len(values) < 2:
        raise ValueError("Dataset has insufficient finite actions.")
    return values.mean(0), values.std(0).clamp_min(1e-6)


def summarize(values):
    values = values.detach().float().flatten().cpu()
    return {
        "mean": float(values.mean()),
        "median": float(values.median()),
        "min": float(values.min()),
        "max": float(values.max()),
        "std": float(values.std(unbiased=False)),
    }


def write_candidate_csv(path, total, interior):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "candidate_index", "planner_expert_path_mse", "planner_expert_interior_mse"
        ])
        writer.writeheader()
        for index, (full, inner) in enumerate(zip(total.cpu(), interior.cpu()), 1):
            writer.writerow({
                "candidate_index": index,
                "planner_expert_path_mse": float(full),
                "planner_expert_interior_mse": float(inner),
            })


def plot(total, interior, summary, output):
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    x = np.arange(1, len(total) + 1)
    axes[0].plot(x, total.cpu(), label="Full path MSE", lw=1.3)
    axes[0].plot(x, interior.cpu(), label="Interior-only MSE", lw=1.3)
    axes[0].scatter(x[int(interior.argmin())], interior.min().cpu(),
                    marker="*", s=100, color="tab:orange", edgecolors="black",
                    linewidths=.5, zorder=3, label="Best interior path")
    axes[0].set(title="Planner paths versus expert latent trajectory",
                xlabel="Planner sample", ylabel="MSE (lower is better)")
    axes[0].grid(alpha=.25)
    axes[0].legend()

    labels = [
        "Planner\ninterior mean", "Planner\ninterior min",
        "LeWM expert-action\ninterior", "LeWM expert-action\nterminal",
        "Inverse dynamics\nnormalized action",
    ]
    values = [
        summary["planner"]["interior"]["mean"],
        summary["planner"]["interior"]["min"],
        summary["lewm_expert_actions"]["interior_mse"],
        summary["lewm_expert_actions"]["terminal_mse"],
        summary["inverse_dynamics"]["normalized_action_mse"],
    ]
    bars = axes[1].bar(labels, values, color=[
        "tab:blue", "tab:blue", "tab:green", "tab:green", "tab:red"
    ])
    axes[1].bar_label(bars, fmt="%.4f", padding=3, rotation=90, fontsize=8)
    axes[1].set(title="Component errors (not all share units)",
                ylabel="MSE (lower is better)")
    axes[1].grid(axis="y", alpha=.25)
    axes[1].tick_params(axis="x", labelrotation=0)
    figure.savefig(output / "component_error_summary.png", dpi=180)
    plt.close(figure)


def main():
    cfg = parse_args()
    if cfg.horizon * cfg.action_block != cfg.goal_offset_steps:
        raise ValueError("goal-offset-steps must equal horizon * action-block.")
    if cfg.planner_samples < 1:
        raise ValueError("planner-samples must be positive.")

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    runtime = LatentPlannerRuntime.from_checkpoint(cfg.planner_checkpoint, device).eval()
    if runtime.action_block != cfg.action_block:
        raise ValueError("action-block differs from checkpoint.")

    dataset = swm.data.HDF5Dataset(
        cfg.dataset_name, frameskip=1, num_steps=1, cache_dir=cfg.cache_dir
    )
    end = cfg.start_step + cfg.goal_offset_steps
    if not 0 <= cfg.episode_index < len(dataset.lengths) or cfg.start_step < 0 or end >= dataset.lengths[cfg.episode_index]:
        raise ValueError("Selected task exceeds its episode.")
    chunk = dataset.load_chunk(
        np.asarray([cfg.episode_index]), np.asarray([cfg.start_step]), np.asarray([end])
    )[0]
    missing = {"pixels", "action"} - set(chunk)
    if missing:
        raise KeyError(f"Dataset task missing {sorted(missing)}")

    preprocessor = get_img_preprocessor("pixels", "pixels", cfg.img_size)
    z_start = image_latent(runtime, chunk["pixels"][0], preprocessor, device)
    z_goal = image_latent(runtime, chunk["pixels"][-1], preprocessor, device)
    # The dataset end is exclusive. Resample all available expert frames to H+1
    # points so z_start/z_goal exactly match the planner conditioning endpoints.
    expert_indices = np.rint(
        np.linspace(0, len(chunk["pixels"]) - 1, cfg.horizon + 1)
    ).astype(np.int64)
    expert_path = torch.cat([
        image_latent(runtime, chunk["pixels"][index], preprocessor, device)[:, None]
        for index in expert_indices
    ], dim=1)

    raw_actions = torch.as_tensor(chunk["action"], dtype=torch.float32)
    required = cfg.horizon * cfg.action_block
    if raw_actions.ndim != 2 or raw_actions.size(0) < required:
        raise ValueError(
            f"Need at least {required} expert actions, got {tuple(raw_actions.shape)}."
        )
    raw_actions = raw_actions[:required]
    action_mean, action_std = finite_action_statistics(dataset)
    normalized_actions = (raw_actions - action_mean) / action_std
    packed_expert_actions = normalized_actions.reshape(
        1, 1, cfg.horizon, cfg.action_block * normalized_actions.size(-1)
    ).to(device)

    generator = torch.Generator(device=device).manual_seed(cfg.noise_seed)
    planner_paths = runtime.sample_paths(
        z_start, z_goal, horizon=cfg.horizon, num_samples=cfg.planner_samples,
        flow_steps=cfg.flow_steps, generator=generator,
    )
    planner_total = (
        planner_paths - expert_path[:, None]
    ).square().mean(dim=(-1, -2))[0]
    planner_interior = (
        planner_paths[:, :, 1:-1] - expert_path[:, None, 1:-1]
    ).square().mean(dim=(-1, -2))[0]

    with torch.no_grad():
        lewm_path = runtime.rollout_paths(
            z_start, packed_expert_actions, cfg.history_size
        )[:, 0]
        lewm_per_state = (lewm_path - expert_path).square().mean(-1)[0]
        inverse_actions = runtime.decode_actions(expert_path[:, None])
        inverse_normalized_mse = (
            inverse_actions - packed_expert_actions
        ).square().mean()
        predicted_raw = (
            inverse_actions[0, 0].reshape(-1, raw_actions.size(-1)).cpu()
            * action_std + action_mean
        )
        inverse_raw_mse = (predicted_raw - raw_actions).square().mean()

    summary = {
        "task": {
            "episode_index": cfg.episode_index,
            "start_step": cfg.start_step,
            "goal_offset_steps": cfg.goal_offset_steps,
            "horizon": cfg.horizon,
            "action_block": cfg.action_block,
            "expert_frame_offsets": expert_indices.tolist(),
        },
        "planner": {
            "sampling": {
                "samples": cfg.planner_samples,
                "flow_steps": cfg.flow_steps,
                "experience_conditioning": False,
            },
            "full_path": summarize(planner_total),
            "interior": summarize(planner_interior),
        },
        "lewm_expert_actions": {
            "full_path_mse": float(lewm_per_state.mean()),
            "interior_mse": float(lewm_per_state[1:-1].mean()),
            "terminal_mse": float(lewm_per_state[-1]),
            "per_state_mse": lewm_per_state.cpu().tolist(),
        },
        "inverse_dynamics": {
            "normalized_action_mse": float(inverse_normalized_mse),
            "raw_action_mse": float(inverse_raw_mse),
        },
    }
    output = cfg.output_dir / (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + f"_ep{cfg.episode_index}_step{cfg.start_step}"
    )
    output.mkdir(parents=True, exist_ok=False)
    write_candidate_csv(output / "planner_path_errors.csv", planner_total, planner_interior)
    plot(planner_total, planner_interior, summary, output)
    metadata = vars(cfg) | {
        "planner_checkpoint": str(cfg.planner_checkpoint),
        "output_dir": str(cfg.output_dir),
        "cache_dir": None if cfg.cache_dir is None else str(cfg.cache_dir),
        "action_normalization": "dataset-wide finite-action z-score, identical to train_latent_planner.py",
    }
    with (output / "component_error_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    with (output / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"analysis_complete output_dir={output}", flush=True)


if __name__ == "__main__":
    main()
