#!/usr/bin/env python3
"""Compute real-rollout MSE and candidate-ranking correlation in one rollout pass."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from analyze_reference_guided_correlation import write_text_summary
from analyze_reference_guided_mse import metric
from reference_guided_real_rollout_common import (
    RealCandidateEvaluator, choose_tasks, parse_args_common,
    pearson, spearman, summary, write_json,
)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_mse_results(output, args, records):
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "candidate_mse.csv", records)
    fields = list(records[0])
    values = {
        key: np.asarray([row[key] for row in records], dtype=np.float64)
        for key in fields[4:]
    }
    report = {
        "settings": vars(args),
        "candidate_metrics": {key: summary(value) for key, value in values.items()},
        "mean_improvement": {
            "endpoint_mse_lewm_minus_guided": float(
                values["lewm_endpoint_mse"].mean() - values["guided_endpoint_mse"].mean()
            ),
            "full_mse_lewm_minus_guided": float(
                values["lewm_full_mse"].mean() - values["guided_full_mse"].mean()
            ),
        },
        "interpretation": (
            "Positive improvement means the reference-guided prediction is closer "
            "to the real simulator rollout."
        ),
    }
    write_json(output / "summary.json", report)

    labels = ["Endpoint MSE", "Full-trajectory MSE"]
    lewm_means = [values["lewm_endpoint_mse"].mean(), values["lewm_full_mse"].mean()]
    guided_means = [values["guided_endpoint_mse"].mean(), values["guided_full_mse"].mean()]
    x = np.arange(2)
    fig, axis = plt.subplots(figsize=(7, 4.5))
    width = 0.36
    axis.bar(x - width / 2, lewm_means, width, label="Autoregressive LeWM", color="#5B8FF9")
    axis.bar(x + width / 2, guided_means, width, label="Reference-guided model", color="#61DDAA")
    axis.set_xticks(x, labels)
    axis.set_ylabel("MSE to simulator-encoded trajectory (lower is better)")
    axis.set_title("Real-rollout trajectory prediction error")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "mse_comparison.png", dpi=200)
    plt.close(fig)
    print(
        f"MSE: endpoint LeWM={lewm_means[0]:.6f}, guided={guided_means[0]:.6f}; "
        f"full LeWM={lewm_means[1]:.6f}, guided={guided_means[1]:.6f}"
    )


def write_correlation_results(output, args, records, task_rows):
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / "candidate_scores.csv", records)
    write_csv(output / "task_ranking_metrics.csv", task_rows)
    candidate = {
        key: np.asarray([row[key] for row in records], dtype=np.float64)
        for key in (
            "simulator_goal_mse",
            "lewm_predicted_goal_mse",
            "guided_predicted_goal_mse",
        )
    }
    report = {
        "settings": vars(args),
        "global_candidate_correlation": {
            "lewm_pearson": pearson(
                candidate["lewm_predicted_goal_mse"], candidate["simulator_goal_mse"]
            ),
            "guided_pearson": pearson(
                candidate["guided_predicted_goal_mse"], candidate["simulator_goal_mse"]
            ),
            "lewm_spearman": spearman(
                candidate["lewm_predicted_goal_mse"], candidate["simulator_goal_mse"]
            ),
            "guided_spearman": spearman(
                candidate["guided_predicted_goal_mse"], candidate["simulator_goal_mse"]
            ),
        },
        "per_task_ranking_metrics": {
            key: summary([row[key] for row in task_rows])
            for key in task_rows[0]
            if key not in ("task_index", "episode", "start_step")
        },
        "interpretation": (
            "Higher Pearson/Spearman and top-1 match are better. Lower selected "
            "true goal MSE and selection regret are better."
        ),
    }
    write_json(output / "summary.json", report)
    write_text_summary(output / "summary.txt", args, task_rows, report)

    limit = np.percentile(
        np.concatenate([
            candidate["simulator_goal_mse"],
            candidate["lewm_predicted_goal_mse"],
            candidate["guided_predicted_goal_mse"],
        ]),
        99,
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
    for axis, key, title, color in (
        (axes[0], "lewm_predicted_goal_mse", "Autoregressive LeWM", "#5B8FF9"),
        (axes[1], "guided_predicted_goal_mse", "Reference-guided model", "#61DDAA"),
    ):
        predicted = candidate[key]
        observed = candidate["simulator_goal_mse"]
        axis.scatter(predicted, observed, s=12, alpha=0.55, color=color, edgecolors="none")
        axis.plot([0, limit], [0, limit], "--", color="black", linewidth=1, label="Ideal: y = x")
        valid = np.isfinite(predicted) & np.isfinite(observed)
        if valid.sum() >= 2 and np.ptp(predicted[valid]) > 0:
            slope, intercept = np.polyfit(predicted[valid], observed[valid], 1)
            x_fit = np.array([0.0, limit])
            axis.plot(
                x_fit, slope * x_fit + intercept, color=color, linewidth=2,
                label=f"Linear fit: y = {slope:.2f}x {intercept:+.2g}",
            )
        axis.set_xlim(0, limit)
        axis.set_ylim(0, limit)
        axis.set_title(title)
        axis.set_xlabel("Predicted terminal-to-goal MSE")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Simulator terminal-to-goal MSE")
    fig.suptitle("Candidate score agreement with real simulator execution")
    fig.tight_layout()
    fig.savefig(output / "goal_score_correlation.png", dpi=200)
    plt.close(fig)
    global_metrics = report["global_candidate_correlation"]
    print(
        f"Global Spearman: LeWM={global_metrics['lewm_spearman']:.4f}, "
        f"guided={global_metrics['guided_spearman']:.4f}"
    )


def main():
    parser = parse_args_common(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    output = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    evaluator = RealCandidateEvaluator(args)
    mse_records, candidate_records, task_rows = [], [], []
    try:
        tasks = choose_tasks(evaluator.dataset, args.tasks, args.goal_offset_steps, args.seed)
        for task_index, (episode, start) in enumerate(tasks):
            data = evaluator.evaluate_task(episode, start)
            lewm_full, lewm_end = metric(data["lewm"], data["target"])
            guided_full, guided_end = metric(data["guided"], data["target"])
            true_score = (
                (data["target"][:, -1] - data["goal"]).square().mean(dim=-1).cpu().numpy()
            )
            lewm_score = (
                (data["lewm"][:, -1] - data["goal"]).square().mean(dim=-1).cpu().numpy()
            )
            guided_score = (
                (data["guided"][:, -1] - data["goal"]).square().mean(dim=-1).cpu().numpy()
            )
            lewm_choice = np.argmin(lewm_score)
            guided_choice = np.argmin(guided_score)
            oracle = np.argmin(true_score)
            task_row = {
                "task_index": task_index, "episode": episode, "start_step": start,
                "lewm_pearson": pearson(lewm_score, true_score),
                "guided_pearson": pearson(guided_score, true_score),
                "lewm_spearman": spearman(lewm_score, true_score),
                "guided_spearman": spearman(guided_score, true_score),
                "lewm_selected_true_goal_mse": float(true_score[lewm_choice]),
                "guided_selected_true_goal_mse": float(true_score[guided_choice]),
                "oracle_true_goal_mse": float(true_score[oracle]),
                "lewm_selection_regret": float(true_score[lewm_choice] - true_score[oracle]),
                "guided_selection_regret": float(true_score[guided_choice] - true_score[oracle]),
                "lewm_top1_match": int(lewm_choice == oracle),
                "guided_top1_match": int(guided_choice == oracle),
            }
            task_rows.append(task_row)
            for candidate in range(args.candidates):
                mse_records.append({
                    "task_index": task_index, "episode": episode, "start_step": start,
                    "candidate": candidate, "lewm_full_mse": lewm_full[candidate],
                    "lewm_endpoint_mse": lewm_end[candidate],
                    "guided_full_mse": guided_full[candidate],
                    "guided_endpoint_mse": guided_end[candidate],
                })
                candidate_records.append({
                    "task_index": task_index, "episode": episode, "start_step": start,
                    "candidate": candidate, "simulator_goal_mse": true_score[candidate],
                    "lewm_predicted_goal_mse": lewm_score[candidate],
                    "guided_predicted_goal_mse": guided_score[candidate],
                })
            print(
                f"task={task_index + 1}/{len(tasks)} episode={episode} start={start} "
                f"LeWM/full={lewm_full.mean():.5f} guided/full={guided_full.mean():.5f} "
                f"rho LeWM={task_row['lewm_spearman']:.3f} "
                f"guided={task_row['guided_spearman']:.3f}",
                flush=True,
            )
    finally:
        evaluator.close()

    write_mse_results(output / "mse", args, mse_records)
    write_correlation_results(output / "correlation", args, candidate_records, task_rows)
    print(f"Saved {len(mse_records)} shared-rollout candidate measurements to {output}")


if __name__ == "__main__":
    main()
