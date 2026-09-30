"""Visualize cross-attention over FIFO experiences during one inference trace."""
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

from latent_planner import LatentPlannerRuntime, load_ltc_components_for_evaluation
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
    parser.add_argument("--flow-steps", type=int, default=16)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--max-memory-size", type=int, default=64)
    parser.add_argument("--noise-seed", type=int, default=20260928)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, default=Path("analysis/cross_attention"))
    return parser.parse_args()


def image_latent(runtime, image, preprocessor, device):
    pixels = torch.as_tensor(image)
    if pixels.ndim == 3 and pixels.shape[-1] in (1, 3):
        pixels = pixels.permute(2, 0, 1)
    if pixels.ndim != 3:
        raise ValueError(f"Expected one image, received {tuple(pixels.shape)}")
    pixels = preprocessor({"pixels": pixels[None]})["pixels"][None].to(device)
    return runtime.lewm.encode({"pixels": pixels})["emb"][:, -1]


class CrossAttentionTracer:
    """Temporarily request per-head weights without changing model parameters."""

    def __init__(self, flow):
        self.flow = flow
        self.calls = []
        self.original = []
        self.round = None
        self.ode_step = None
        self.costs = None

    def set_context(self, round_index, ode_step, costs):
        self.round = int(round_index)
        self.ode_step = int(ode_step)
        self.costs = costs.detach().float().cpu().numpy().reshape(-1).copy()

    def __enter__(self):
        for layer, block in enumerate(self.flow.blocks):
            module = block.cross_attn
            original = module.forward

            def wrapped(query, key, value, *args, _original=original,
                        _layer=layer, **kwargs):
                # The production caller requests no weights. This changes only
                # the returned diagnostic tensor, not the attention output.
                kwargs["need_weights"] = True
                kwargs["average_attn_weights"] = False
                output, weights = _original(query, key, value, *args, **kwargs)
                if self.round is None or self.costs is None:
                    raise RuntimeError("Cross-attention call has no trace context.")
                # weights: [candidate_batch, heads, query_tokens, memory_entries]
                attention = weights.detach().mean(dim=(0, 1, 2)).cpu().numpy()
                self.calls.append({
                    "round": self.round,
                    "ode_step": self.ode_step,
                    "layer": _layer,
                    "costs": self.costs.copy(),
                    "attention": attention,
                })
                return output, weights

            module.forward = wrapped
            self.original.append((module, original))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for module, original in self.original:
            module.forward = original
        self.original.clear()


@torch.no_grad()
def sample(runtime, z_start, z_goal, noise, flow_steps,
           path_features=None, path_costs=None, tracer=None, round_index=None):
    if (path_features is None) != (path_costs is None):
        raise ValueError("features and costs must be supplied together")
    batch, candidates, _, latent_dim = noise.shape
    x = noise.reshape(batch * candidates, noise.size(2), latent_dim).clone()
    start = z_start[:, None].expand(-1, candidates, -1).reshape(batch * candidates, -1)
    goal = z_goal[:, None].expand(-1, candidates, -1).reshape(batch * candidates, -1)
    features = None if path_features is None else path_features.repeat_interleave(candidates, 0)
    costs = None if path_costs is None else path_costs.repeat_interleave(candidates, 0)
    dt = 1.0 / max(flow_steps, 1)
    for ode_step in range(flow_steps):
        if tracer is not None:
            tracer.set_context(round_index, ode_step, path_costs)
        t = torch.full((batch * candidates,), ode_step * dt, device=x.device, dtype=x.dtype)
        x = x + runtime.flow(x, t, start, goal, features, costs) * dt
    return torch.cat([start[:, None], x, goal[:, None]], 1).reshape(
        batch, candidates, x.size(1) + 2, latent_dim
    )


def rows_from_calls(calls):
    rows = []
    for call in calls:
        size = len(call["costs"])
        for index, (cost, attention) in enumerate(zip(call["costs"], call["attention"])):
            rows.append({
                "round": call["round"],
                "ode_step": call["ode_step"],
                "layer": call["layer"],
                "memory_index": index,
                "memory_age": size - 1 - index,
                "memory_size": size,
                "experience_cost": float(cost),
                "attention_weight": float(attention),
            })
    return rows


def save_rows(rows, output):
    with (output / "cross_attention.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot_heatmaps(calls, output, depth):
    for round_index in sorted({call["round"] for call in calls}):
        selected = [call for call in calls if call["round"] == round_index]
        memory_size = len(selected[0]["costs"])
        flow_steps = 1 + max(call["ode_step"] for call in selected)
        figure, axes = plt.subplots(1, depth, figsize=(4.4 * depth, 4.4),
                                    squeeze=False, constrained_layout=True)
        for layer in range(depth):
            matrix = np.full((flow_steps, memory_size), np.nan)
            for call in selected:
                if call["layer"] == layer:
                    matrix[call["ode_step"]] = call["attention"]
            image = axes[0, layer].imshow(matrix, origin="lower", aspect="auto",
                                          cmap="magma", vmin=0)
            axes[0, layer].set(
                title=f"Layer {layer}", xlabel="FIFO index (oldest to newest)",
                ylabel="ODE step",
            )
            figure.colorbar(image, ax=axes[0, layer], label="Mean weight")
        figure.suptitle(
            f"Round {round_index}; memory costs oldest to newest: "
            f"{np.array2string(selected[0]['costs'], precision=3)}",
            fontsize=10,
        )
        figure.savefig(output / f"round_{round_index:02d}_attention_heatmaps.png", dpi=180)
        plt.close(figure)


def plot_summary(rows, output):
    summary = []
    rounds = sorted({row["round"] for row in rows})
    for round_index in rounds:
        for index in sorted({r["memory_index"] for r in rows if r["round"] == round_index}):
            selected = [r for r in rows if r["round"] == round_index and r["memory_index"] == index]
            summary.append({
                "round": round_index,
                "memory_index": index,
                "memory_age": selected[0]["memory_age"],
                "cost": float(np.mean([r["experience_cost"] for r in selected])),
                "attention": float(np.mean([r["attention_weight"] for r in selected])),
            })

    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    points = axes[0].scatter(
        [r["cost"] for r in summary], [r["attention"] for r in summary],
        c=[r["round"] for r in summary], cmap="viridis", s=45, alpha=.85,
    )
    axes[0].set(title="Attention versus experience cost",
                xlabel="LTC cost (lower is better)", ylabel="Mean attention weight")
    axes[0].grid(alpha=.25)
    figure.colorbar(points, ax=axes[0], label="Inference round")

    for round_index in rounds:
        selected = [r for r in summary if r["round"] == round_index]
        axes[1].plot([r["memory_age"] for r in selected],
                     [r["attention"] for r in selected],
                     marker="o", ms=3, label=f"Round {round_index}")
    axes[1].set(title="Attention by FIFO age", xlabel="FIFO age (0 = newest)",
                ylabel="Mean attention weight")
    axes[1].grid(alpha=.25)
    axes[1].legend(ncol=2, fontsize=8)
    figure.savefig(output / "cross_attention_summary.png", dpi=180)
    plt.close(figure)

    costs = np.array([r["cost"] for r in summary])
    attention = np.array([r["attention"] for r in summary])
    pearson = float(np.corrcoef(costs, attention)[0, 1]) if len(summary) > 1 else float("nan")
    return summary, pearson


def main():
    cfg = parse_args()
    if cfg.horizon * cfg.action_block != cfg.goal_offset_steps:
        raise ValueError("goal-offset-steps must equal horizon * action-block.")
    if cfg.candidates < 1 or cfg.rounds < 2 or cfg.max_memory_size < 1:
        raise ValueError("candidates and max-memory-size must be positive; rounds >= 2.")
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    payload = torch.load(cfg.planner_checkpoint, map_location="cpu", weights_only=False)
    runtime = LatentPlannerRuntime.from_checkpoint(cfg.planner_checkpoint, device).eval()
    ltc = payload.get("experience", {}).get("ltc")
    if ltc is None:
        raise ValueError("planner checkpoint has no bundled experience.ltc")
    encoder, scorer = load_ltc_components_for_evaluation(
        ltc, latent_dim=runtime.flow.latent_dim, device=device
    )
    if encoder.representation_dim != runtime.flow.path_feature_dim:
        raise ValueError("LTC and planner feature dimensions differ.")
    if runtime.action_block != cfg.action_block:
        raise ValueError("action-block differs from checkpoint.")

    dataset = swm.data.HDF5Dataset(cfg.dataset_name, frameskip=1,
                                   num_steps=1, cache_dir=cfg.cache_dir)
    end = cfg.start_step + cfg.goal_offset_steps
    if not 0 <= cfg.episode_index < len(dataset.lengths) or cfg.start_step < 0 or end >= dataset.lengths[cfg.episode_index]:
        raise ValueError("Selected task exceeds its episode.")
    chunk = dataset.load_chunk(np.asarray([cfg.episode_index]),
                               np.asarray([cfg.start_step]), np.asarray([end]))[0]
    if "pixels" not in chunk:
        raise KeyError("Dataset task is missing pixels.")
    preprocessor = get_img_preprocessor("pixels", "pixels", cfg.img_size)
    z_start = image_latent(runtime, chunk["pixels"][0], preprocessor, device)
    z_goal = image_latent(runtime, chunk["pixels"][-1], preprocessor, device)
    generator = torch.Generator(device=device).manual_seed(cfg.noise_seed)
    noise = torch.randn(1, cfg.candidates * cfg.rounds, cfg.horizon - 1,
                        runtime.flow.latent_dim, generator=generator, device=device)

    memory_feature = memory_cost = None
    cursor = 0
    with CrossAttentionTracer(runtime.flow) as tracer:
        for round_index in range(cfg.rounds):
            current_noise = noise[:, cursor:cursor + cfg.candidates]
            paths = sample(
                runtime, z_start, z_goal, current_noise, cfg.flow_steps,
                memory_feature, memory_cost, tracer if memory_feature is not None else None,
                round_index,
            )
            flat_paths = paths.flatten(0, 1)
            goals = z_goal[:, None].expand(-1, cfg.candidates, -1).reshape(
                -1, z_goal.size(-1)
            )
            features = encoder(flat_paths, goals).reshape(1, cfg.candidates, -1)
            costs = scorer(features.flatten(0, 1)).reshape(1, cfg.candidates)
            memory_feature = features if memory_feature is None else torch.cat([memory_feature, features], 1)
            memory_cost = costs if memory_cost is None else torch.cat([memory_cost, costs], 1)
            memory_feature = memory_feature[:, -cfg.max_memory_size:]
            memory_cost = memory_cost[:, -cfg.max_memory_size:]
            cursor += cfg.candidates
            print(f"round={round_index} memory_after={memory_cost.size(1)} "
                  f"cost_mean={memory_cost.mean().item():.5f} "
                  f"cost_std={memory_cost.std(unbiased=False).item():.5f}", flush=True)

    if not tracer.calls:
        raise RuntimeError("No cross-attention was recorded.")
    output = cfg.output_dir / (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + f"_ep{cfg.episode_index}_step{cfg.start_step}_{cfg.candidates}x{cfg.rounds}"
    )
    output.mkdir(parents=True, exist_ok=False)
    rows = rows_from_calls(tracer.calls)
    save_rows(rows, output)
    plot_heatmaps(tracer.calls, output, runtime.flow.depth)
    summary, pearson = plot_summary(rows, output)
    metadata = vars(cfg) | {
        "planner_checkpoint": str(cfg.planner_checkpoint),
        "output_dir": str(cfg.output_dir),
        "cache_dir": None if cfg.cache_dir is None else str(cfg.cache_dir),
        "attention_weight_definition": "Mean over candidates, heads, and query tokens. Each layer/ODE-step sums to one over all FIFO experiences.",
        "cost_attention_pearson": pearson,
        "traced_rounds": sorted({c["round"] for c in tracer.calls}),
        "summary_entry_count": len(summary),
    }
    with (output / "metadata.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"analysis_complete output_dir={output}", flush=True)


if __name__ == "__main__":
    main()
