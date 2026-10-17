#!/usr/bin/env python3
"""Compare autoregressive LeWM and reference-guided trajectory prediction to PushT rollouts."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from reference_guided_real_rollout_common import (
    RealCandidateEvaluator, choose_tasks, parse_args_common, summary, write_json,
)


def metric(prediction, target):
    difference = (prediction - target).square()
    return difference.mean(dim=(1, 2)).cpu().numpy(), difference[:, -1].mean(dim=1).cpu().numpy()


def main():
    parser = parse_args_common(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    output = Path(args.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)

    evaluator = RealCandidateEvaluator(args)
    records = []
    try:
        tasks = choose_tasks(evaluator.dataset, args.tasks, args.goal_offset_steps, args.seed)
        for task_index, (episode, start) in enumerate(tasks):
            data = evaluator.evaluate_task(episode, start)
            lewm_full, lewm_end = metric(data["lewm"], data["target"])
            guided_full, guided_end = metric(data["guided"], data["target"])
            for candidate in range(args.candidates):
                records.append({
                    "task_index": task_index, "episode": episode, "start_step": start,
                    "candidate": candidate, "lewm_full_mse": lewm_full[candidate],
                    "lewm_endpoint_mse": lewm_end[candidate],
                    "guided_full_mse": guided_full[candidate],
                    "guided_endpoint_mse": guided_end[candidate],
                })
            print(
                f"task={task_index + 1}/{len(tasks)} episode={episode} start={start} "
                f"lewm_end={lewm_end.mean():.5f} guided_end={guided_end.mean():.5f} "
                f"lewm_full={lewm_full.mean():.5f} guided_full={guided_full.mean():.5f}",
                flush=True,
            )
    finally:
        evaluator.close()

    fields = list(records[0])
    with (output / "candidate_mse.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    values = {key: np.asarray([row[key] for row in records], dtype=np.float64) for key in fields[4:]}
    report = {
        "settings": vars(args),
        "candidate_metrics": {key: summary(value) for key, value in values.items()},
        "mean_improvement": {
            "endpoint_mse_lewm_minus_guided": float(values["lewm_endpoint_mse"].mean() - values["guided_endpoint_mse"].mean()),
            "full_mse_lewm_minus_guided": float(values["lewm_full_mse"].mean() - values["guided_full_mse"].mean()),
        },
        "interpretation": "Positive improvement means the reference-guided prediction is closer to the real simulator rollout.",
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

    print(f"\nSaved {len(records)} candidate measurements to {output}")
    print(
        "aggregate "
        f"endpoint: LeWM={lewm_means[0]:.6f}, guided={guided_means[0]:.6f}; "
        f"full: LeWM={lewm_means[1]:.6f}, guided={guided_means[1]:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
