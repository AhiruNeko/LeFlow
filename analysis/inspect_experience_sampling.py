"""Analyze raw latent trajectories produced by real experience-guided sampling.

For each fixed dataset start/goal, this runs the same candidates-times-rounds
loop as inference: a round samples N candidates using the current FIFO, then
encodes/scores and appends all N generated paths.  The visualized points are
the actual raw latent trajectories generated at selected pre-round FIFO sizes.
No expert, mismatch, or synthetic-noise path enters the experience bank.
"""
from __future__ import annotations
import argparse, csv, json, math, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from analysis.inspect_ltc_distribution import load_ltc, make_loader
from latent_planner import LatentPlannerRuntime


def parse_sizes(value):
    sizes = sorted({int(x.strip()) for x in value.split(",") if x.strip()})
    if not sizes or sizes[0] < 0:
        raise ValueError("--memory-sizes must be non-negative")
    return sizes


def encode_score(paths, goal, encoder, cost_model):
    batch, count, length, latent = paths.shape
    features = encoder(
        paths.reshape(batch * count, length, latent),
        goal[:, None].expand(-1, count, -1).reshape(batch * count, latent),
    ).reshape(batch, count, -1)
    return features, cost_model(features.flatten(0, 1)).reshape(batch, count)


def raw_path_vector(paths):
    """Flatten intermediate raw latents; fixed start/goal do not dominate PCA."""
    if paths.size(2) < 3:
        raise ValueError("a path needs start, intermediate states, and goal")
    return paths[:, :, 1:-1].flatten(start_dim=2)


def project_pca(feature):
    centered = feature - feature.mean(dim=0, keepdim=True)
    _, singular, vectors = torch.pca_lowrank(
        centered, q=min(3, centered.size(0) - 1, centered.size(1))
    )
    points = centered @ vectors[:, :3]
    points = torch.nn.functional.pad(points, (0, max(0, 3 - points.size(1))))
    return points, singular.square() / centered.square().sum().clamp_min(1e-12)


@torch.no_grad()
def sample_candidate_rounds(runtime, encoder, cost_model, z_start, z_goal, args, sizes, generator):
    """Run real candidates x rounds sampling and retain selected pre-round sets."""
    records = {}
    paths = features = costs = None

    for round_index in range(args.rounds):
        memory_size = 0 if paths is None else paths.size(1)
        generated = runtime.sample_paths(
            z_start,
            z_goal,
            horizon=args.horizon,
            num_samples=args.candidates_per_round,
            flow_steps=args.flow_steps,
            path_features=features,
            path_costs=costs,
            generator=generator,
        )
        generated_features, generated_costs = encode_score(
            generated, z_goal, encoder, cost_model
        )

        # These are actual candidates drawn under this FIFO condition.
        if memory_size in sizes and memory_size not in records:
            records[memory_size] = (
                generated.clone(),
                generated_costs.clone(),
                round_index + 1,
            )

        paths = generated if paths is None else torch.cat((paths, generated), dim=1)
        features = (
            generated_features
            if features is None
            else torch.cat((features, generated_features), dim=1)
        )
        costs = (
            generated_costs
            if costs is None
            else torch.cat((costs, generated_costs), dim=1)
        )
        if paths.size(1) > args.experience_max_size:
            paths = paths[:, -args.experience_max_size:]
            features = features[:, -args.experience_max_size:]
            costs = costs[:, -args.experience_max_size:]

    missing = [size for size in sizes if size not in records]
    if missing:
        raise ValueError(
            "The requested FIFO sizes were never observed before a generation "
            f"round: {missing}. Increase --rounds, use compatible sizes, or "
            "lower --candidates-per-round."
        )
    return records


def save_pca_plot(rows, path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable: skipped PCA PNG")
        return
    tasks = sorted({row["task_id"] for row in rows})
    sizes = sorted({row["memory_size"] for row in rows})
    columns = min(4, len(tasks))
    figure, axes = plt.subplots(
        math.ceil(len(tasks) / columns),
        columns,
        figsize=(4.2 * columns, 3.7 * math.ceil(len(tasks) / columns)),
    )
    axes = list(getattr(axes, "flat", [axes]))
    palette = plt.cm.viridis
    for axis, task in zip(axes, tasks):
        for index, size in enumerate(sizes):
            points = [
                row
                for row in rows
                if row["task_id"] == task and row["memory_size"] == size
            ]
            axis.scatter(
                [row["pc1"] for row in points],
                [row["pc2"] for row in points],
                s=15,
                alpha=.6,
                color=palette(index / max(len(sizes) - 1, 1)),
                label=f"memory={size}",
            )
        axis.set(
            title=f"Task {task}: fixed start / goal",
            xlabel="Raw-latent PCA-1",
            ylabel="Raw-latent PCA-2",
        )
        axis.grid(alpha=.18)
    for axis in axes[len(tasks):]:
        axis.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, labels, ncol=min(5, len(sizes)), loc="upper center", frameon=False
    )
    figure.suptitle(
        "Raw model-sampled trajectories under real FIFO experience sizes", y=1.02
    )
    figure.tight_layout()
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)
    print(f"raw_latent_pca_png={path}")


def save_umap(rows, task_centered_raw, path, args):
    try:
        import umap
    except ImportError:
        print("umap-learn unavailable: skipped UMAP artifacts")
        return
    embedding = umap.UMAP(
        n_components=3,
        n_neighbors=min(
            max(2, args.umap_neighbors), task_centered_raw.size(0) - 1
        ),
        min_dist=args.umap_min_dist,
        metric="euclidean",
        random_state=args.umap_random_state,
    ).fit_transform(task_centered_raw.float().numpy())
    for row, point in zip(rows, embedding):
        row.update(
            {"umap1": float(point[0]), "umap2": float(point[1]), "umap3": float(point[2])}
        )

    csv_path = path.with_name(path.stem + "_umap3d.csv")
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    try:
        import matplotlib.pyplot as plt
        figure = plt.figure(figsize=(8, 6))
        axis = figure.add_subplot(111, projection="3d")
        sizes = sorted({row["memory_size"] for row in rows})
        palette = plt.cm.viridis
        for index, size in enumerate(sizes):
            points = [row for row in rows if row["memory_size"] == size]
            axis.scatter(
                [row["umap1"] for row in points],
                [row["umap2"] for row in points],
                [row["umap3"] for row in points],
                s=12,
                alpha=.6,
                color=palette(index / max(len(sizes) - 1, 1)),
                label=f"memory={size}",
            )
        axis.set(
            xlabel="UMAP-1",
            ylabel="UMAP-2",
            zlabel="UMAP-3",
            title="Task-centered raw latent trajectories: real experience FIFO",
        )
        axis.legend(markerscale=1.5, fontsize=8)
        figure.tight_layout()
        png_path = path.with_name(path.stem + "_umap3d.png")
        figure.savefig(png_path, dpi=190)
        plt.close(figure)
        print(f"raw_latent_umap_png={png_path}")
    except ImportError:
        print("matplotlib unavailable: skipped UMAP PNG")

    try:
        import plotly.express as px
    except ImportError:
        print("plotly unavailable: skipped interactive UMAP HTML")
        return
    figure = px.scatter_3d(
        rows,
        x="umap1",
        y="umap2",
        z="umap3",
        color="memory_size_label",
        symbol="task_label",
        hover_data={
            "task_id": True,
            "memory_size": True,
            "round": True,
            "candidate_id": True,
            "predicted_cost": ":.4f",
        },
        title="Task-centered raw latent trajectories: real experience FIFO",
    )
    figure.update_traces(marker={"size": 3, "opacity": .6})
    figure.update_layout(scene={"aspectmode": "data"})
    html_path = path.with_name(path.stem + "_umap3d.html")
    figure.write_html(html_path, include_plotlyjs=True, full_html=True)
    print(f"raw_latent_umap_html={html_path} source_csv={csv_path}")


def save_quality_probe_plot(rows, path):
    """Plot cost and encoder-feature displacement by known path quality."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable: skipped quality-probe PNG")
        return

    conditions = list(dict.fromkeys(row["condition"] for row in rows))
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    palette = plt.cm.tab20
    for index, condition in enumerate(conditions):
        points = [row for row in rows if row["condition"] == condition]
        color = palette(index % 20)
        axes[0].scatter(
            [row["feature_pc1"] for row in points],
            [row["feature_pc2"] for row in points],
            s=16, alpha=.65, color=color, label=condition,
        )
        axes[1].scatter(
            [row["severity"] for row in points],
            [row["predicted_cost"] for row in points],
            s=16, alpha=.65, color=color, label=condition,
        )
        axes[2].scatter(
            [row["severity"] for row in points],
            [row["feature_l2_to_expert"] for row in points],
            s=16, alpha=.65, color=color, label=condition,
        )
    axes[0].set(title="Encoder feature PCA", xlabel="Feature PC1", ylabel="Feature PC2")
    axes[1].set(title="Predicted cost by perturbation", xlabel="Perturbation scale", ylabel="Cost (lower is better)")
    axes[2].set(title="Feature displacement from expert", xlabel="Perturbation scale", ylabel="L2 distance")
    for axis in axes:
        axis.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=min(5, len(conditions)), frameon=False)
    figure.tight_layout(rect=(0, 0, 1, .88))
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)
    print(f"quality_probe_png={path}")


@torch.no_grad()
def inspect_quality_probe(path, goal, encoder, cost_model, args, generator):
    """Compare known expert, noisy, and goal-mismatched path quality levels."""
    if path.size(0) < 2:
        raise ValueError("quality probe requires at least two tasks for goal mismatch")
    scales = [float(value) for value in args.quality_noise_scales.split(",")]
    if not scales or any(scale <= 0 for scale in scales):
        raise ValueError("--quality-noise-scales must be comma-separated positive values")
    if any(b <= a for a, b in zip(scales[:-1], scales[1:])):
        raise ValueError("--quality-noise-scales must be strictly increasing")

    latent_scale = path.std(unbiased=False).clamp_min(1e-6)
    all_noise = torch.randn(path.shape, device=path.device, dtype=path.dtype, generator=generator)
    all_noise[:, 0] = 0
    interior_noise = torch.randn(path.shape, device=path.device, dtype=path.dtype, generator=generator)
    interior_noise[:, 0] = 0
    interior_noise[:, -1] = 0

    conditions = [("expert", "expert", 0.0, path, goal)]
    for scale in scales:
        conditions.append((
            f"noise_all_{scale:g}", "noise_all_after_start", scale,
            path + scale * latent_scale * all_noise, goal,
        ))
        conditions.append((
            f"noise_interior_{scale:g}", "noise_interior_only", scale,
            path + scale * latent_scale * interior_noise, goal,
        ))
    conditions.append((
        "goal_mismatch", "goal_mismatch", float(max(scales)), path,
        goal.roll(shifts=1, dims=0),
    ))

    expert_feature = encoder(path, goal)
    feature_blocks, metadata = [], []
    for name, kind, severity, candidate_path, candidate_goal in conditions:
        feature = encoder(candidate_path, candidate_goal)
        cost = cost_model(feature)
        feature_blocks.append(feature)
        metadata.extend((name, kind, severity, index, cost[index], feature[index]) for index in range(path.size(0)))

    joined_feature = torch.cat(feature_blocks, dim=0).cpu()
    pca, explained = project_pca(joined_feature)
    rows, cursor = [], 0
    summaries = {}
    for name, kind, severity, task, cost, feature in metadata:
        reference = expert_feature[task]
        cosine = torch.nn.functional.cosine_similarity(feature[None], reference[None]).item()
        distance = (feature - reference).norm().item()
        rows.append({
            "condition": name,
            "quality_type": kind,
            "severity": severity,
            "task_id": task,
            "predicted_cost": float(cost),
            "cost_minus_expert": float(cost - cost_model(expert_feature)[task]),
            "feature_cosine_to_expert": cosine,
            "feature_l2_to_expert": distance,
            "feature_pc1": float(pca[cursor, 0]),
            "feature_pc2": float(pca[cursor, 1]),
            "feature_pc3": float(pca[cursor, 2]),
        })
        cursor += 1
    for name, _, _, _, _ in conditions:
        points = [row for row in rows if row["condition"] == name]
        summaries[name] = {
            "count": len(points),
            "cost_mean": sum(row["predicted_cost"] for row in points) / len(points),
            "cost_std": float(torch.tensor([row["predicted_cost"] for row in points]).std(unbiased=False)),
            "cost_minus_expert_mean": sum(row["cost_minus_expert"] for row in points) / len(points),
            "feature_cosine_to_expert_mean": sum(row["feature_cosine_to_expert"] for row in points) / len(points),
            "feature_l2_to_expert_mean": sum(row["feature_l2_to_expert"] for row in points) / len(points),
        }
    return rows, summaries, [float(value) for value in explained]

@torch.no_grad()
def run(args):
    if args.candidates_per_round < 1:
        raise ValueError("--candidates-per-round must be positive")
    if args.experience_max_size < args.candidates_per_round:
        raise ValueError("--experience-max-size must fit one candidate round")
    sizes = parse_sizes(args.memory_sizes)
    if max(sizes) > args.experience_max_size:
        raise ValueError("a requested memory size exceeds --experience-max-size")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    runtime = LatentPlannerRuntime.from_checkpoint(args.planner_checkpoint, device=device)
    runtime.eval().requires_grad_(False)

    path = runtime.encode_pixels(next(iter(make_loader(args)))["pixels"].to(device))
    path = path[:args.num_tasks]
    if path.size(1) != args.horizon + 1:
        raise ValueError("dataset path must contain horizon + 1 latent states")
    z_start, z_goal = path[:, 0], path[:, -1]
    encoder, cost_model = load_ltc(
        args.ltc_checkpoint,
        latent_dim=path.size(-1),
        device=device,
    )
    if args.horizon > encoder.max_horizon:
        raise ValueError("LTC encoder max_horizon is smaller than --horizon")

    quality_rows, quality_summary, quality_explained = inspect_quality_probe(
        path, z_goal, encoder, cost_model, args, generator
    )

    records = sample_candidate_rounds(
        runtime, encoder, cost_model, z_start, z_goal, args, sizes, generator
    )
    raw, costs, rounds = {}, {}, {}
    for size, (sampled_paths, sampled_costs, round_index) in records.items():
        raw[size] = raw_path_vector(sampled_paths).cpu()
        costs[size] = sampled_costs.cpu()
        rounds[size] = round_index

    baseline = {task: raw[0][task].mean(dim=0) for task in range(path.size(0))}
    all_raw, centered_raw, labels, per_task = [], [], [], {}
    for task in range(path.size(0)):
        per_task[str(task)] = {}
        for size in sizes:
            vectors = raw[size][task]
            predicted = costs[size][task]
            centroid = vectors.mean(dim=0)
            per_task[str(task)][str(size)] = {
                "predicted_cost_mean": float(predicted.mean()),
                "predicted_cost_std": float(predicted.std(unbiased=False)),
                "within_raw_latent_rms": float(
                    (vectors - centroid).norm(dim=-1).mean()
                ),
                "raw_latent_centroid_l2_to_memory_0": float(
                    (centroid - baseline[task]).norm()
                ),
                "generated_round": rounds[size],
            }
            all_raw.append(vectors)
            centered_raw.append(vectors - baseline[task])
            labels.extend(
                (task, size, rounds[size], candidate, float(predicted[candidate]))
                for candidate in range(vectors.size(0))
            )

    joined_raw = torch.cat(all_raw)
    task_centered_raw = torch.cat(centered_raw)
    raw_pca, raw_explained = project_pca(joined_raw)
    centered_pca, centered_explained = project_pca(task_centered_raw)
    rows = [
        {
            "task_id": task,
            "task_label": f"task={task}",
            "memory_size": size,
            "memory_size_label": f"memory={size}",
            "round": round_index,
            "candidate_id": candidate,
            "predicted_cost": predicted_cost,
            "pc1": float(raw_pca[index, 0]),
            "pc2": float(raw_pca[index, 1]),
            "pc3": float(raw_pca[index, 2]),
            "centered_pc1": float(centered_pca[index, 0]),
            "centered_pc2": float(centered_pca[index, 1]),
            "centered_pc3": float(centered_pca[index, 2]),
        }
        for index, (task, size, round_index, candidate, predicted_cost) in enumerate(labels)
    ]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    quality_csv_path = output.with_name(output.stem + "_quality_probe.csv")
    with quality_csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(quality_rows[0]))
        writer.writeheader()
        writer.writerows(quality_rows)
    quality_plot_path = output.with_name(output.stem + "_quality_probe.png")
    save_quality_probe_plot(quality_rows, quality_plot_path)

    csv_path = output.with_name(output.stem + "_raw_latent_pca.csv")
    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "description": (
            "One fixed start/goal per task. Every visible point is a raw latent "
            "trajectory generated during a candidates-times-rounds FIFO loop."
        ),
        "memory_source": "autoregressive_real_flow_samples_only",
        "visualization_representation": (
            "flattened intermediate raw latent states; start and goal omitted "
            "because they are fixed within each task"
        ),
        "checkpoints": {
            "planner": args.planner_checkpoint,
            "ltc_checkpoint": args.ltc_checkpoint,
        },
        "dataset": args.dataset,
        "seed": args.seed,
        "horizon": args.horizon,
        "action_block": args.action_block,
        "flow_steps": args.flow_steps,
        "num_tasks": path.size(0),
        "candidates_per_round": args.candidates_per_round,
        "rounds": args.rounds,
        "experience_max_size": args.experience_max_size,
        "memory_sizes": sizes,
        "raw_latent_pca_explained_variance": [float(x) for x in raw_explained],
        "task_centered_raw_latent_pca_explained_variance": [
            float(x) for x in centered_explained
        ],
        "per_task": per_task,
        "quality_probe": {
            "description": (
                "Known-quality expert, multi-scale noisy, and goal-mismatched paths "
                "encoded by the current trajectory encoder and scored by the current cost model."
            ),
            "noise_scales": [float(value) for value in args.quality_noise_scales.split(",")],
            "feature_pca_explained_variance": quality_explained,
            "conditions": quality_summary,
            "csv": str(quality_csv_path),
            "png": str(quality_plot_path),
        },
        "artifacts": {
            "raw_latent_pca_csv": str(csv_path),
            "raw_latent_pca_by_task_png": str(
                output.with_name(output.stem + "_raw_latent_pca_by_task.png")
            ),
        },
    }
    output.write_text(json.dumps(report, indent=2))
    save_pca_plot(rows, output.with_name(output.stem + "_raw_latent_pca_by_task.png"))
    save_umap(
        rows,
        task_centered_raw,
        output.with_name(output.stem + "_task_centered_raw_latent"),
        args,
    )
    print(f"summary_written={output}")
    for size in sizes:
        mean_cost = sum(
            per_task[str(task)][str(size)]["predicted_cost_mean"]
            for task in range(path.size(0))
        ) / path.size(0)
        shift = sum(
            per_task[str(task)][str(size)]["raw_latent_centroid_l2_to_memory_0"]
            for task in range(path.size(0))
        ) / path.size(0)
        print(
            f"memory_size={size:>2} round={rounds[size]:>2} "
            f"candidate_cost_mean={mean_cost:.4f} raw_centroid_shift={shift:.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner-checkpoint", required=True)
    parser.add_argument("--ltc-checkpoint", required=True)
    parser.add_argument("--dataset", default="pusht_expert_train")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--num-tasks", type=int, default=8)
    parser.add_argument("--candidates-per-round", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=17)
    parser.add_argument("--experience-max-size", type=int, default=64)
    parser.add_argument("--flow-steps", type=int, default=16)
    parser.add_argument("--memory-sizes", default="0,4,16,64")
    parser.add_argument(
        "--quality-noise-scales",
        default="0.02,0.05,0.1,0.2",
        help="Positive, increasing noise scales for expert-path quality probing.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--umap-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=.1)
    parser.add_argument("--umap-random-state", type=int, default=42)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
