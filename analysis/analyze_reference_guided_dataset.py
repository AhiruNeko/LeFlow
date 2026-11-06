#!/usr/bin/env python3
"""Evaluate direct latent prediction on sampled dataset clips, without candidate ranking."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from reference_guided_real_rollout_common import (
    RealCandidateEvaluator, TASK_PROFILES, choose_tasks, summary, write_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=sorted(TASK_PROFILES), required=True,
                        help="Dataset/environment profile only.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--cache-dir", default=None,
                        help="Optional HDF5Dataset read cache, not a rollout cache.")
    parser.add_argument("--planner-checkpoint", required=True,
                        help="Supplies one reference path per clip.")
    parser.add_argument("--reference-guided-checkpoint", required=True)
    parser.add_argument("--lewm-checkpoint", required=True)
    parser.add_argument("--samples", type=int, default=500,
                        help="Number of distinct dataset clips.")
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


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def direct_mse(predicted, actual):
    error = np.square(predicted - actual)
    return float(error.mean()), float(error[-1].mean())


def dimension_metrics(predicted, actual):
    """Feature-wise Pearson and R² across observations, not across latent features."""
    if predicted.shape != actual.shape or predicted.ndim != 2:
        raise ValueError("Expected paired [observations, latent_dim] matrices")
    x = predicted.astype(np.float64)
    y = actual.astype(np.float64)
    x_centered = x - x.mean(axis=0)
    y_centered = y - y.mean(axis=0)
    x_ss = np.square(x_centered).sum(axis=0)
    y_ss = np.square(y_centered).sum(axis=0)
    covariance = (x_centered * y_centered).sum(axis=0)
    denominator = np.sqrt(x_ss * y_ss)
    valid_pearson = denominator > 1e-12
    valid_r2 = y_ss > 1e-12
    pearson_values = np.full(y.shape[1], np.nan)
    r2_values = np.full(y.shape[1], np.nan)
    pearson_values[valid_pearson] = covariance[valid_pearson] / denominator[valid_pearson]
    squared_error = np.square(x - y).sum(axis=0)
    r2_values[valid_r2] = 1 - squared_error[valid_r2] / y_ss[valid_r2]
    return {
        "pearson": pearson_values,
        "r2": r2_values,
        "mse": np.square(x - y).mean(axis=0),
        "true_variance": np.var(y, axis=0),
        "valid_pearson": valid_pearson,
        "valid_r2": valid_r2,
    }


def finite_summary(values):
    valid = np.asarray(values, dtype=np.float64)
    valid = valid[np.isfinite(valid)]
    return {
        "valid_dimensions": int(valid.size),
        "mean": float(valid.mean()) if valid.size else None,
        "median": float(np.median(valid)) if valid.size else None,
        "q25": float(np.percentile(valid, 25)) if valid.size else None,
        "q75": float(np.percentile(valid, 75)) if valid.size else None,
    }


def plot_mse(path, rows):
    x = np.arange(2)
    fig, axis = plt.subplots(figsize=(7, 4.5))
    for offset, model, color in (
        (-0.18, "lewm", "#5B8FF9"), (0.18, "guided", "#61DDAA")
    ):
        values = [
            np.mean([row[f"{model}_endpoint_mse"] for row in rows]),
            np.mean([row[f"{model}_full_mse"] for row in rows]),
        ]
        axis.bar(x + offset, values, 0.36, color=color,
                 label="Continuous LeWM" if model == "lewm" else "L2GWM")
    axis.set_xticks(x, ["Endpoint", "Full trajectory"])
    axis.set_ylabel("MSE to executed latent (lower is better)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".pdf"), dpi=200)
    plt.close(fig)


def plot_dimension_metrics(path, results):
    fig, axis = plt.subplots(figsize=(7, 5.3))
    lewm, guided = results["lewm"], results["guided"]
    paired = (
        np.isfinite(lewm["pearson"]) & np.isfinite(lewm["r2"])
        & np.isfinite(guided["pearson"]) & np.isfinite(guided["r2"])
    )
    all_r2 = np.concatenate([lewm["r2"][paired], guided["r2"][paired]])
    if all_r2.size:
        lower = min(-0.1, float(np.percentile(all_r2, 2)) - 0.1)
        lower = max(lower, -5.0)
        for i in np.flatnonzero(paired):
            axis.plot(
                [lewm["pearson"][i], guided["pearson"][i]],
                [max(lewm["r2"][i], lower), max(guided["r2"][i], lower)],
                color="#B8BCC2", alpha=0.35, linewidth=0.6, zorder=1,
            )
        for model, color, label in (
            ("lewm", "#5B8FF9", "Continuous LeWM"),
            ("guided", "#008B72", "L2GWM"),
        ):
            values = results[model]
            axis.scatter(values["pearson"][paired], np.maximum(values["r2"][paired], lower),
                         s=18, alpha=0.75, color=color, label=label, zorder=2)
        axis.set_ylim(lower, 1.05)
        clipped = int(np.sum(all_r2 < lower))
        if clipped:
            axis.text(0.02, 0.02, f"{clipped} R² values clipped below {lower:.2f}; see CSV",
                      transform=axis.transAxes, fontsize=8)
    else:
        axis.text(0.5, 0.5, "No dimensions with defined Pearson and R²",
                  ha="center", va="center", transform=axis.transAxes)
    axis.axvline(0, color="black", linewidth=0.7, alpha=0.5)
    axis.axhline(0, color="black", linewidth=0.7, alpha=0.5)
    axis.set_xlim(-1.05, 1.05)
    axis.set_xlabel("Pearson across samples, same latent dimension (higher is better)")
    axis.set_ylabel("R² across samples, same latent dimension (higher is better)")
    axis.set_title("Endpoint prediction by latent dimension")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".pdf"), dpi=200)
    plt.close(fig)


def plot_fitted_latents(path, endpoint, seed):
    """Use ground-truth per-feature statistics for both predictions and targets."""
    actual = endpoint["actual"].astype(np.float64)
    mean = actual.mean(axis=0)
    scale = actual.std(axis=0)
    valid = scale > 1e-6
    if not np.any(valid):
        raise ValueError("All endpoint latent dimensions are nearly constant")
    y = ((actual[:, valid] - mean[valid]) / scale[valid]).ravel()
    predictions = {
        model: ((endpoint[model][:, valid] - mean[valid]) / scale[valid]).ravel()
        for model in ("lewm", "guided")
    }
    indices = np.random.default_rng(seed).choice(
        len(y), size=min(len(y), 20000), replace=False
    )
    displayed = np.concatenate(
        [y[indices], predictions["lewm"][indices], predictions["guided"][indices]]
    )
    lower, upper = np.percentile(displayed, [0.5, 99.5])
    bound = max(abs(float(lower)), abs(float(upper)), 1.0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
    for axis, model, title, line_color, cloud_color in (
        (axes[0], "lewm", "Continuous LeWM", "#3265A8", "#8FAED3"),
        (axes[1], "guided", "L2GWM", "#008B72", "#8CCABC"),
    ):
        x = predictions[model]
        axis.scatter(x[indices], y[indices], s=2.0, alpha=0.18,
                     color=cloud_color, edgecolors="none", rasterized=True)
        axis.plot([-bound, bound], [-bound, bound], "--", color="black",
                  linewidth=1.2, label="Ideal: y = x")
        if np.ptp(x) > 0:
            slope, intercept = np.polyfit(x, y, 1)
            line_x = np.asarray([-bound, bound])
            axis.plot(line_x, slope * line_x + intercept, color=line_color, linewidth=1.7,
                      label=f"Fit: y = {slope:.2f}x {intercept:+.2f}")
        axis.set_xlim(-bound, bound)
        axis.set_ylim(-bound, bound)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("Predicted endpoint latent (true-feature standardized)")
        axis.set_title(title)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Executed endpoint latent (true-feature standardized)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".pdf"), dpi=200)
    plt.close(fig)
    return int(valid.sum())


def main():
    args = parse_args()
    output = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    mse_dir, correlation_dir = output / "mse", output / "correlation"
    mse_dir.mkdir(parents=True, exist_ok=False)
    correlation_dir.mkdir(parents=True, exist_ok=False)
    evaluator = RealCandidateEvaluator(args)
    rows = []
    endpoint_values = {"actual": [], "lewm": [], "guided": []}
    full_values = {"actual": [], "lewm": [], "guided": []}
    try:
        clips = choose_tasks(evaluator.dataset, args.samples, args.goal_offset_steps, args.seed)
        for index, (episode, start) in enumerate(clips):
            result = evaluator.evaluate_task(episode, start)
            actual = result["target"][0].cpu().numpy()
            lewm = result["lewm"][0].cpu().numpy()
            guided = result["guided"][0].cpu().numpy()
            if actual.shape != lewm.shape or actual.shape != guided.shape:
                raise ValueError("Predicted and executed trajectories must have identical shapes")
            lewm_full, lewm_end = direct_mse(lewm, actual)
            guided_full, guided_end = direct_mse(guided, actual)
            rows.append({
                "sample_index": index, "episode": episode, "start_step": start,
                "lewm_full_mse": lewm_full, "lewm_endpoint_mse": lewm_end,
                "guided_full_mse": guided_full, "guided_endpoint_mse": guided_end,
            })
            for key, value in (("actual", actual), ("lewm", lewm), ("guided", guided)):
                endpoint_values[key].append(value[-1])
                full_values[key].append(value)
            print(
                f"sample={index + 1}/{len(clips)} episode={episode} start={start} "
                f"endpoint_mse LeWM={lewm_end:.5f} guided={guided_end:.5f}",
                flush=True,
            )
    finally:
        evaluator.close()

    write_csv(mse_dir / "sample_mse.csv", rows)
    mse_keys = ("lewm_full_mse", "lewm_endpoint_mse", "guided_full_mse", "guided_endpoint_mse")
    mse_report = {
        "settings": vars(args),
        "model_display_names": {"lewm": "LeWM", "guided": "L2GWM"},
        "sample_metrics": {key: summary([row[key] for row in rows]) for key in mse_keys},
    }
    mse_report["mean_improvement"] = {
        "endpoint_mse_lewm_minus_guided":
            mse_report["sample_metrics"]["lewm_endpoint_mse"]["mean"]
            - mse_report["sample_metrics"]["guided_endpoint_mse"]["mean"],
        "full_mse_lewm_minus_guided":
            mse_report["sample_metrics"]["lewm_full_mse"]["mean"]
            - mse_report["sample_metrics"]["guided_full_mse"]["mean"],
    }
    write_json(mse_dir / "summary.json", mse_report)
    plot_mse(mse_dir / "mse_comparison.png", rows)

    endpoint = {key: np.stack(value) for key, value in endpoint_values.items()}
    full = {key: np.concatenate(value, axis=0) for key, value in full_values.items()}
    scopes = {
        "endpoint_latent": endpoint,
        "full_trajectory_latent": full,
    }
    correlation_report = {
        "settings": vars(args),
        "model_display_names": {"lewm": "LeWM", "guided": "L2GWM"},
        "method": "For each aligned latent dimension, Pearson and R² are computed across samples (or sample-step observations for the full trajectory).",
        "endpoint_latent": {},
        "full_trajectory_latent": {},
    }
    endpoint_results = {}
    dimension_rows = []
    for scope, matrices in scopes.items():
        actual = matrices["actual"]
        for model in ("lewm", "guided"):
            metrics = dimension_metrics(matrices[model], actual)
            if scope == "endpoint_latent":
                endpoint_results[model] = metrics
            correlation_report[scope][model] = {
                "pearson": finite_summary(metrics["pearson"]),
                "r2": finite_summary(metrics["r2"]),
                "mse": summary(metrics["mse"]),
            }
            for dim in range(actual.shape[1]):
                dimension_rows.append({
                    "scope": scope,
                    "model": model,
                    "dimension": dim,
                    "pearson": float(metrics["pearson"][dim]) if metrics["valid_pearson"][dim] else None,
                    "r2": float(metrics["r2"][dim]) if metrics["valid_r2"][dim] else None,
                    "mse": float(metrics["mse"][dim]),
                    "true_variance": float(metrics["true_variance"][dim]),
                })
    write_csv(correlation_dir / "dimension_metrics.csv", dimension_rows)
    standardized_dimensions = plot_fitted_latents(
        correlation_dir / "latent_correlation.png", endpoint, args.seed
    )
    plot_dimension_metrics(correlation_dir / "dimension_correlation_r2.png", endpoint_results)
    correlation_report["standardized_plot_dimensions"] = standardized_dimensions
    write_json(correlation_dir / "summary.json", correlation_report)
    lines = [
        "Direct latent prediction versus simulator execution",
        "LeWM vs L2GWM (the world-model component of L2G-LeFlow)",
        f"Samples: {len(rows)}",
        "Pearson and R²: same latent dimension across samples; undefined low-variance dimensions excluded.",
        f"Standardized scatter: {standardized_dimensions} dimensions with nontrivial true variance.",
        "",
    ]
    for scope in scopes:
        lines.append(scope)
        for model in ("lewm", "guided"):
            values = correlation_report[scope][model]
            lines.append(
                f"  {'LeWM' if model == 'lewm' else 'L2GWM'}: Pearson median={values['pearson']['median']} "
                f"(valid={values['pearson']['valid_dimensions']}); "
                f"R² median={values['r2']['median']} "
                f"(valid={values['r2']['valid_dimensions']})"
            )
    (correlation_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {len(rows)} direct-prediction samples to {output}")


if __name__ == "__main__":
    main()
