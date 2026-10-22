"""Three-way MSE and goal-score correlation on identical simulator candidates."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ablation.model import AblationWorldModel
from analysis.reference_guided_real_rollout_common import (
    TASK_PROFILES, choose_tasks, make_dataset, parse_args_common,
    pearson, spearman, summary, write_json,
)
from latent_planner import LatentPlannerRuntime
from reference_guided_world_model import (
    ReferenceGuidedWorldModel, continuous_lewm_predictions, parallel_segment_predictions,
)
from stable_worldmodel_compat import create_world
from train_reference_guided_world_model import image_transform, set_random_seed, simulator_targets

MODELS = ("lewm", "reference_guided", "ablation")


class ThreeWayEvaluator:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        self.runtime = LatentPlannerRuntime.from_checkpoint(
            args.planner_checkpoint, device=self.device, lewm_checkpoint=args.lewm_checkpoint
        ).requires_grad_(False).eval()
        self.guided = ReferenceGuidedWorldModel.from_checkpoint(
            args.reference_guided_checkpoint, device=self.device
        ).requires_grad_(False).eval()
        self.ablation = AblationWorldModel.from_checkpoint(
            args.ablation_checkpoint, device=self.device
        ).requires_grad_(False).eval()
        payload = torch.load(args.planner_checkpoint, map_location="cpu", weights_only=False)
        self.horizon = int(payload["config"]["planner"]["horizon"])
        for model in (self.guided, self.ablation):
            arch = model.architecture
            if arch.latent_dim != self.runtime.flow.latent_dim or arch.action_dim != self.runtime.inverse_dynamics.action_dim:
                raise ValueError("Compared models must match the planner/IDM dimensions.")
            if arch.max_horizon < self.horizon:
                raise ValueError("Compared model cannot represent the planner horizon.")
        self.profile = TASK_PROFILES[args.task]
        self.dataset = make_dataset(args.dataset_name, args.cache_dir, args.task)
        actions = torch.as_tensor(self.dataset.get_col_data("action"), dtype=torch.float32)
        actions = actions[torch.isfinite(actions).all(dim=1)]
        if not len(actions):
            raise ValueError("Dataset has no finite actions for normalization.")
        self.action_mean = actions.mean(0).to(self.device)
        self.action_std = actions.std(0).clamp_min(1e-6).to(self.device)
        self.transform = image_transform(args.img_size)
        self.world = create_world(
            dict(self.profile["world"], num_envs=args.candidates,
                 max_episode_steps=self.horizon * self.runtime.action_block),
            image_shape=(args.img_size, args.img_size),
        )
        self.generator = torch.Generator(device=self.device).manual_seed(args.seed)

    @torch.no_grad()
    def evaluate_task(self, episode, start):
        # Preserve the original analysis clip/goal selection and simulator API.
        raw = self.dataset.load_chunk([episode], [start], [start + self.args.goal_offset_steps])[0]
        pixels = torch.as_tensor(raw["pixels"])
        start_latent = self.runtime.encode_pixels(self.transform(pixels[:1]).to(self.device))[:, 0]
        goal = self.runtime.encode_pixels(self.transform(pixels[-1:]).to(self.device))[:, 0]
        paths = self.runtime.sample_paths(
            start_latent, goal, horizon=self.horizon, num_samples=self.args.candidates,
            flow_steps=self.args.flow_steps, generator=self.generator,
        )
        actions = self.runtime.decode_actions(paths)
        target = simulator_targets(
            self.runtime, self.world, self.dataset, np.asarray([episode]), np.asarray([start]),
            paths, actions, goal_offset=self.args.goal_offset_steps,
            image_tx=self.transform, action_mean=self.action_mean, action_std=self.action_std,
            device=self.device, task_restore=self.profile["simulator_restore"],
        )
        path, action = paths[0], actions[0]
        local = parallel_segment_predictions(self.runtime.lewm, path, action)
        return {
            "target": target, "goal": goal.expand(self.args.candidates, -1),
            "lewm": continuous_lewm_predictions(
                self.runtime.lewm, path, action, history_size=self.args.history_size
            ),
            "reference_guided": self.guided(path, action, local)["trajectory"],
            "ablation": self.ablation(path, action)["trajectory"],
        }

    def close(self):
        self.world.close()


def measure_task(data, task_index, episode, start):
    truth = (data["target"][:, -1] - data["goal"]).square().mean(-1).cpu().numpy()
    count = len(truth)
    candidates = [dict(task_index=task_index, episode=episode, start_step=start,
                       candidate=i, simulator_goal_mse=float(truth[i])) for i in range(count)]
    task = dict(task_index=task_index, episode=episode, start_step=start)
    oracle = int(np.argmin(truth))
    for name in MODELS:
        diff = (data[name] - data["target"]).square()
        full = diff.mean((1, 2)).cpu().numpy()
        final = diff[:, -1].mean(-1).cpu().numpy()
        scores = (data[name][:, -1] - data["goal"]).square().mean(-1).cpu().numpy()
        choice = int(np.argmin(scores))
        task.update({
            f"{name}_pearson": pearson(scores, truth),
            f"{name}_spearman": spearman(scores, truth),
            f"{name}_selection_regret": float(truth[choice] - truth[oracle]),
            f"{name}_selected_true_goal_mse": float(truth[choice]),
            f"{name}_top1_match": int(choice == oracle),
        })
        for i, row in enumerate(candidates):
            row.update({f"{name}_full_mse": float(full[i]),
                        f"{name}_final_mse": float(final[i]),
                        f"{name}_goal_mse": float(scores[i])})
    task["oracle_true_goal_mse"] = float(truth[oracle])
    return candidates, task


def save_reports(output, args, records, task_rows):
    for filename, rows in (("candidate_metrics.csv", records), ("task_ranking_metrics.csv", task_rows)):
        with (output / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    values = {key: np.asarray([row[key] for row in records], dtype=float)
              for key in records[0] if key not in ("task_index", "episode", "start_step", "candidate")}
    correlations = {}
    for name in MODELS:
        correlations[name] = {
            "pearson": pearson(values[f"{name}_goal_mse"], values["simulator_goal_mse"]),
            "spearman": spearman(values[f"{name}_goal_mse"], values["simulator_goal_mse"]),
        }
    report = {
        "settings": vars(args),
        "candidate_metrics": {key: summary(value) for key, value in values.items()},
        "global_goal_error_correlation": correlations,
        "per_task_ranking": {key: summary([row[key] for row in task_rows])
                             for key in task_rows[0] if key not in ("task_index", "episode", "start_step")},
        "ablation_advantage": {
            f"{other}_minus_ablation_{part}_mse": float(
                values[f"{other}_{part}_mse"].mean() - values[f"ablation_{part}_mse"].mean()
            ) for other in ("lewm", "reference_guided") for part in ("full", "final")
        },
        "interpretation": "All three models share each candidate and simulator rollout. Positive advantage favors the ablation. LeWM is continuous autoregressive inference.",
    }
    write_json(output / "summary.json", report)
    x = np.arange(2)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, name in enumerate(MODELS):
        ax.bar(x + (i - 1) * 0.25,
               [values[f"{name}_{part}_mse"].mean() for part in ("full", "final")],
               width=0.25, label=name)
    ax.set_xticks(x, ["Full trajectory", "Endpoint"])
    ax.set_ylabel("MSE to simulator rollout")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "mse_comparison.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=True, sharey=True)
    for ax, name in zip(axes, MODELS):
        ax.scatter(values[f"{name}_goal_mse"], values["simulator_goal_mse"], s=10, alpha=0.5)
        ax.set_title(name)
        ax.set_xlabel("Predicted terminal-to-goal MSE")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Simulator terminal-to-goal MSE")
    fig.tight_layout()
    fig.savefig(output / "goal_score_correlation.png", dpi=160)
    plt.close(fig)
    print(json_report(report), flush=True)


def json_report(report):
    return " | ".join(
        f"{name}: full={report['candidate_metrics'][name + '_full_mse']['mean']:.6f}, "
        f"final={report['candidate_metrics'][name + '_final_mse']['mean']:.6f}, "
        f"Pearson={report['global_goal_error_correlation'][name]['pearson']:.4f}"
        for name in MODELS
    )


def main():
    parser = parse_args_common(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--ablation-checkpoint", required=True)
    args = parser.parse_args()
    if args.tasks < 1 or args.candidates < 2:
        parser.error("Use at least one task and two candidates.")
    root = Path(__file__).resolve().parent
    output = Path(args.output_dir).expanduser().resolve()
    if not output.is_relative_to(root):
        parser.error("Analysis output must stay inside ablation/.")
    output = output / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    set_random_seed(args.seed)
    evaluator = ThreeWayEvaluator(args)
    records, tasks = [], []
    try:
        for i, (episode, start) in enumerate(choose_tasks(
            evaluator.dataset, args.tasks, args.goal_offset_steps, args.seed
        )):
            candidates, task = measure_task(evaluator.evaluate_task(episode, start), i, episode, start)
            records.extend(candidates)
            tasks.append(task)
            print(f"task={i + 1}/{args.tasks} episode={episode} start={start}", flush=True)
    finally:
        evaluator.close()
    save_reports(output, args, records, tasks)
    print(f"Saved MSE and correlation analysis to {output}", flush=True)


if __name__ == "__main__":
    main()
