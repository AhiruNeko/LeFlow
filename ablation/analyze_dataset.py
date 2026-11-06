#!/usr/bin/env python3
"""Four-model direct latent prediction on sampled clips; no candidate ranking."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ablation.model import AblationWorldModel
from analysis.reference_guided_real_rollout_common import (
    RealCandidateEvaluator, TASK_PROFILES, choose_tasks, summary, write_json,
)
from reference_guided_world_model import continuous_lewm_predictions, parallel_segment_predictions
from train_reference_guided_world_model import set_random_seed, simulator_targets

# Reuse the existing direct-prediction metric definitions, rather than the
# candidate-ranking metrics in ablation/analyze.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from analyze_reference_guided_dataset import (
    dimension_metrics, direct_mse, finite_summary, write_csv,
)

MODELS = (
    ("lewm", "Continuous LeWM"),
    ("guided", "L2GWM"),
    ("no_local", "No local"),
    ("no_local_no_path", "No local + no planner path"),
)
COLORS = ("#5B8FF9", "#008B72", "#E59A28", "#9667B1")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(TASK_PROFILES), required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--cache-dir", default=None, help="Optional HDF5 read cache, not rollout cache")
    parser.add_argument("--planner-checkpoint", required=True)
    parser.add_argument("--lewm-checkpoint", required=True)
    parser.add_argument("--reference-guided-checkpoint", required=True)
    parser.add_argument("--no-local-checkpoint", required=True)
    parser.add_argument("--no-local-no-path-checkpoint", required=True)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--goal-offset-steps", type=int, default=50)
    parser.add_argument("--flow-steps", type=int, default=16)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.samples < 2 or args.goal_offset_steps < 1:
        parser.error("--samples must be at least 2 and --goal-offset-steps positive")
    args.candidates = 1
    return args


class FourModelEvaluator(RealCandidateEvaluator):
    def __init__(self, args):
        super().__init__(args)
        self.no_local = AblationWorldModel.from_checkpoint(
            args.no_local_checkpoint, device=self.device
        ).requires_grad_(False).eval()
        self.no_local_no_path = AblationWorldModel.from_checkpoint(
            args.no_local_no_path_checkpoint, device=self.device
        ).requires_grad_(False).eval()
        for model, variant in ((self.no_local, "no_local"),
                               (self.no_local_no_path, "endpoints_actions")):
            if model.variant != variant:
                raise ValueError(f"Expected {variant!r} checkpoint, got {model.variant!r}")
        for model in (self.guided, self.no_local, self.no_local_no_path):
            arch = model.architecture
            if (arch.latent_dim != self.runtime.flow.latent_dim
                    or arch.action_dim != self.runtime.inverse_dynamics.action_dim
                    or arch.max_horizon < self.horizon):
                raise ValueError("Model architecture does not match planner/IDM or horizon")

    @torch.no_grad()
    def evaluate_task(self, episode, start):
        raw = self.dataset.load_chunk(
            [int(episode)], [int(start)], [int(start) + int(self.args.goal_offset_steps)]
        )[0]
        pixels = torch.as_tensor(raw["pixels"])
        if pixels.ndim != 4:
            raise ValueError(f"Expected clip pixels [T,C,H,W], got {tuple(pixels.shape)}")
        z_start = self.runtime.encode_pixels(self.transform(pixels[:1]).to(self.device))[:, 0]
        z_goal = self.runtime.encode_pixels(self.transform(pixels[-1:]).to(self.device))[:, 0]
        paths = self.runtime.sample_paths(
            z_start, z_goal, horizon=self.horizon, num_samples=1,
            flow_steps=int(self.args.flow_steps), generator=self.generator,
        )
        actions = self.runtime.decode_actions(paths)
        target = simulator_targets(
            self.runtime, self.world, self.rollout_dataset,
            np.asarray([episode]), np.asarray([start]), paths, actions,
            goal_offset=int(self.args.goal_offset_steps), image_tx=self.transform,
            action_mean=self.action_mean, action_std=self.action_std,
            device=self.device, task_restore=self.task_restore,
        )
        path, action = paths[0], actions[0]
        local = parallel_segment_predictions(self.runtime.lewm, path, action)
        return {
            "actual": target[0],
            "lewm": continuous_lewm_predictions(
                self.runtime.lewm, path, action, history_size=int(self.args.history_size)
            ),
            "guided": self.guided(path, action, local)["trajectory"],
            "no_local": self.no_local(path, action)["trajectory"],
            "no_local_no_path": self.no_local_no_path(path, action)["trajectory"],
        }


def save_plot(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".pdf"), dpi=200)
    plt.close(fig)


def plot_mse(path, rows):
    fig, axis = plt.subplots(figsize=(9, 4.8))
    x, width = np.arange(2), 0.2
    for i, (key, label) in enumerate(MODELS):
        values = [np.mean([row[f"{key}_{part}_mse"] for row in rows])
                  for part in ("endpoint", "full")]
        axis.bar(x + (i - 1.5) * width, values, width, label=label, color=COLORS[i])
    axis.set_xticks(x, ["Endpoint", "Full trajectory"])
    axis.set_ylabel("MSE to executed latent (lower is better)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    save_plot(fig, path)


def plot_dimensions(path, results):
    fig, axis = plt.subplots(figsize=(8, 5.3))
    for i, (key, label) in enumerate(MODELS):
        metric = results[key]
        valid = np.isfinite(metric["pearson"]) & np.isfinite(metric["r2"])
        axis.scatter(metric["pearson"][valid], metric["r2"][valid], s=14,
                     alpha=0.65, color=COLORS[i], label=label)
    axis.axvline(0, color="black", linewidth=0.7, alpha=0.5)
    axis.axhline(0, color="black", linewidth=0.7, alpha=0.5)
    axis.set_xlim(-1.05, 1.05)
    axis.set_xlabel("Pearson across samples, same latent dimension")
    axis.set_ylabel("R² across samples, same latent dimension")
    axis.set_title("Endpoint prediction by latent dimension")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    save_plot(fig, path)


def plot_fitted(path, endpoint, seed):
    actual = endpoint["actual"].astype(np.float64)
    mean, scale = actual.mean(axis=0), actual.std(axis=0)
    valid = scale > 1e-6
    if not np.any(valid):
        raise ValueError("All endpoint latent dimensions are nearly constant")
    y = ((actual[:, valid] - mean[valid]) / scale[valid]).ravel()
    predictions = {
        key: ((endpoint[key][:, valid] - mean[valid]) / scale[valid]).ravel()
        for key, _ in MODELS
    }
    indices = np.random.default_rng(seed).choice(len(y), size=min(len(y), 20000), replace=False)
    displayed = np.concatenate([y[indices]] + [x[indices] for x in predictions.values()])
    lower, upper = np.percentile(displayed, [0.5, 99.5])
    bound = max(abs(float(lower)), abs(float(upper)), 1.0)
    fig, axes = plt.subplots(2, 2, figsize=(10, 9), sharex=True, sharey=True)
    for axis, (key, label), color in zip(axes.flat, MODELS, COLORS):
        x = predictions[key]
        axis.scatter(x[indices], y[indices], s=2, alpha=0.18,
                     color=color, edgecolors="none", rasterized=True)
        axis.plot([-bound, bound], [-bound, bound], "--", color="black",
                  linewidth=1.0, label="Ideal: y = x")
        if np.ptp(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            line_x = np.array([-bound, bound])
            axis.plot(line_x, slope * line_x + intercept, color=color, linewidth=1.5,
                      label=f"Fit: y = {slope:.2f}x {intercept:+.2f}")
        axis.set_xlim(-bound, bound)
        axis.set_ylim(-bound, bound)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(label)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    fig.supxlabel("Predicted endpoint latent (true-feature standardized)")
    fig.supylabel("Executed endpoint latent (true-feature standardized)")
    save_plot(fig, path)
    return int(valid.sum())


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    output = Path(args.output_dir).expanduser().resolve()
    if not output.is_relative_to(root):
        raise ValueError("Analysis output must stay inside ablation/")
    output /= datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    mse_dir, correlation_dir = output / "mse", output / "correlation"
    mse_dir.mkdir(parents=True, exist_ok=False)
    correlation_dir.mkdir(parents=True, exist_ok=False)
    set_random_seed(args.seed)
    evaluator = FourModelEvaluator(args)
    rows = []
    endpoint_values = {key: [] for key in ("actual", *(name for name, _ in MODELS))}
    full_values = {key: [] for key in endpoint_values}
    try:
        clips = choose_tasks(evaluator.dataset, args.samples, args.goal_offset_steps, args.seed)
        for index, (episode, start) in enumerate(clips):
            result = evaluator.evaluate_task(episode, start)
            actual = result["actual"].cpu().numpy()
            row = {"sample_index": index, "episode": episode, "start_step": start}
            for key in endpoint_values:
                value = result[key].cpu().numpy()
                if value.shape != actual.shape:
                    raise ValueError(f"{key} trajectory shape does not match simulator target")
                endpoint_values[key].append(value[-1])
                full_values[key].append(value)
                if key != "actual":
                    row[f"{key}_full_mse"], row[f"{key}_endpoint_mse"] = direct_mse(value, actual)
            rows.append(row)
            print(f"sample={index + 1}/{len(clips)} episode={episode} start={start}", flush=True)
    finally:
        evaluator.close()

    write_csv(mse_dir / "sample_mse.csv", rows)
    mse_keys = [f"{key}_{part}_mse" for key, _ in MODELS for part in ("full", "endpoint")]
    write_json(mse_dir / "summary.json", {
        "settings": vars(args), "model_display_names": dict(MODELS),
        "sample_metrics": {key: summary([row[key] for row in rows]) for key in mse_keys},
        "method": "One planner/IDM path per sampled clip, no candidate ranking. MSE is against simulator-executed encoded latent trajectory.",
    })
    plot_mse(mse_dir / "mse_comparison.png", rows)

    endpoint = {key: np.stack(value) for key, value in endpoint_values.items()}
    full = {key: np.concatenate(value, axis=0) for key, value in full_values.items()}
    scopes = {"endpoint_latent": endpoint, "full_trajectory_latent": full}
    report = {
        "settings": vars(args), "model_display_names": dict(MODELS),
        "method": "Pearson and R² per aligned latent dimension across samples (or sample-step observations), then summarized across dimensions.",
    }
    dimension_rows, endpoint_results = [], {}
    for scope, matrices in scopes.items():
        actual = matrices["actual"]
        report[scope] = {}
        for key, _ in MODELS:
            metrics = dimension_metrics(matrices[key], actual)
            if scope == "endpoint_latent":
                endpoint_results[key] = metrics
            report[scope][key] = {
                "pearson": finite_summary(metrics["pearson"]),
                "r2": finite_summary(metrics["r2"]),
                "mse": summary(metrics["mse"]),
            }
            for dim in range(actual.shape[1]):
                dimension_rows.append({
                    "scope": scope, "model": key, "dimension": dim,
                    "pearson": float(metrics["pearson"][dim]) if metrics["valid_pearson"][dim] else None,
                    "r2": float(metrics["r2"][dim]) if metrics["valid_r2"][dim] else None,
                    "mse": float(metrics["mse"][dim]),
                    "true_variance": float(metrics["true_variance"][dim]),
                })
    write_csv(correlation_dir / "dimension_metrics.csv", dimension_rows)
    report["standardized_plot_dimensions"] = plot_fitted(
        correlation_dir / "latent_correlation.png", endpoint, args.seed
    )
    plot_dimensions(correlation_dir / "dimension_correlation_r2.png", endpoint_results)
    write_json(correlation_dir / "summary.json", report)
    lines = [
        "Direct latent prediction versus simulator execution (one path per clip; no candidate ranking)",
        f"Samples: {len(rows)}",
        "Pearson and R²: same latent dimension across samples; undefined low-variance dimensions excluded.",
        f"Standardized scatter: {report['standardized_plot_dimensions']} dimensions with nontrivial true variance.",
        "",
    ]
    for scope in scopes:
        lines.append(scope)
        for key, label in MODELS:
            values = report[scope][key]
            lines.append(
                f"  {label}: Pearson median={values['pearson']['median']} "
                f"(valid={values['pearson']['valid_dimensions']}); "
                f"R² median={values['r2']['median']} "
                f"(valid={values['r2']['valid_dimensions']})"
            )
    (correlation_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {len(rows)} direct-prediction samples to {output}")


if __name__ == "__main__":
    main()
