#!/usr/bin/env python3
"""Measure whether candidate scores from LeWM or the reference-guided model match simulator ranking."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from reference_guided_real_rollout_common import (
    RealCandidateEvaluator, choose_tasks, parse_args_common, pearson, spearman, summary, write_json,
)


def main():
    parser = parse_args_common(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    output = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)

    evaluator = RealCandidateEvaluator(args)
    records, task_rows = [], []
    try:
        tasks = choose_tasks(evaluator.dataset, args.tasks, args.goal_offset_steps, args.seed)
        for task_index, (episode, start) in enumerate(tasks):
            data = evaluator.evaluate_task(episode, start)
            true_score = (data["target"][:, -1] - data["goal"]).square().mean(dim=-1).cpu().numpy()
            lewm_score = (data["lewm"][:, -1] - data["goal"]).square().mean(dim=-1).cpu().numpy()
            guided_score = (data["guided"][:, -1] - data["goal"]).square().mean(dim=-1).cpu().numpy()

            lewm_choice, guided_choice, oracle = np.argmin(lewm_score), np.argmin(guided_score), np.argmin(true_score)
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
                records.append({
                    "task_index": task_index, "episode": episode, "start_step": start,
                    "candidate": candidate, "simulator_goal_mse": true_score[candidate],
                    "lewm_predicted_goal_mse": lewm_score[candidate],
                    "guided_predicted_goal_mse": guided_score[candidate],
                })
            print(
                f"task={task_index + 1}/{len(tasks)} "
                f"rho LeWM={task_row['lewm_spearman']:.3f} guided={task_row['guided_spearman']:.3f}; "
                f"regret LeWM={task_row['lewm_selection_regret']:.5f} guided={task_row['guided_selection_regret']:.5f}",
                flush=True,
            )
    finally:
        evaluator.close()

    for filename, rows in (("candidate_scores.csv", records), ("task_ranking_metrics.csv", task_rows)):
        with (output / filename).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    candidate = {key: np.asarray([row[key] for row in records], dtype=np.float64) for key in ("simulator_goal_mse", "lewm_predicted_goal_mse", "guided_predicted_goal_mse")}
    report = {
        "settings": vars(args),
        "global_candidate_correlation": {
            "lewm_pearson": pearson(candidate["lewm_predicted_goal_mse"], candidate["simulator_goal_mse"]),
            "guided_pearson": pearson(candidate["guided_predicted_goal_mse"], candidate["simulator_goal_mse"]),
            "lewm_spearman": spearman(candidate["lewm_predicted_goal_mse"], candidate["simulator_goal_mse"]),
            "guided_spearman": spearman(candidate["guided_predicted_goal_mse"], candidate["simulator_goal_mse"]),
        },
        "per_task_ranking_metrics": {key: summary([row[key] for row in task_rows]) for key in task_rows[0] if key not in ("task_index", "episode", "start_step")},
        "interpretation": "Higher Pearson/Spearman and top-1 match are better. Lower selected true goal MSE and selection regret are better.",
    }
    write_json(output / "summary.json", report)

    limit = np.percentile(np.concatenate([candidate["simulator_goal_mse"], candidate["lewm_predicted_goal_mse"], candidate["guided_predicted_goal_mse"]]), 99)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
    for axis, key, title, color in (
        (axes[0], "lewm_predicted_goal_mse", "Autoregressive LeWM", "#5B8FF9"),
        (axes[1], "guided_predicted_goal_mse", "Reference-guided model", "#61DDAA"),
    ):
        axis.scatter(candidate[key], candidate["simulator_goal_mse"], s=12, alpha=0.55, color=color, edgecolors="none")
        axis.plot([0, limit], [0, limit], "--", color="black", linewidth=1)
        axis.set_xlim(0, limit)
        axis.set_ylim(0, limit)
        axis.set_title(title)
        axis.set_xlabel("Predicted terminal-to-goal MSE")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Simulator terminal-to-goal MSE")
    fig.suptitle("Candidate score agreement with real simulator execution")
    fig.tight_layout()
    fig.savefig(output / "goal_score_correlation.png", dpi=200)
    plt.close(fig)

    print(f"\nSaved {len(records)} candidate scores and {len(task_rows)} task summaries to {output}")
    print(
        "global Spearman: "
        f"LeWM={report['global_candidate_correlation']['lewm_spearman']:.4f}, "
        f"guided={report['global_candidate_correlation']['guided_spearman']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
