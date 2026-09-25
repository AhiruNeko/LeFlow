"""Inspect the cost and representation geometry of a trained LTC checkpoint."""
from __future__ import annotations

import argparse
import csv
import math
import json
import sys
from pathlib import Path

# This script lives under analysis/, while the model modules live at project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import stable_pretraining as spt
import stable_worldmodel as swm
import torch

from latent_planner import load_lewm, load_ltc_components_for_evaluation
from latent_trajectory_cost import TrajectoryCostModel, TrajectoryEncoder
from utils import get_img_preprocessor



def load_ltc(checkpoint: str, latent_dim: int, device: torch.device):
    """Load a bundled LTC checkpoint."""
    return load_ltc_components_for_evaluation(
        checkpoint,
        latent_dim=latent_dim,
        device=device,
    )


def make_loader(args: argparse.Namespace):
    dataset = swm.data.HDF5Dataset(
        name=args.dataset,
        frameskip=args.action_block,
        num_steps=args.horizon + 1,
        keys_to_load=["pixels"],
        keys_to_cache=[],
        transform=None,
    )
    dataset.transform = spt.data.transforms.Compose(
        get_img_preprocessor(source="pixels", target="pixels", img_size=args.img_size)
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
        drop_last=True,
    )


def quantiles(values: torch.Tensor) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "p05": float(torch.quantile(values, 0.05)),
        "p50": float(torch.quantile(values, 0.50)),
        "p95": float(torch.quantile(values, 0.95)),
    }


@torch.no_grad()
def inspect_mode(
    *,
    mode: str,
    path: torch.Tensor,
    goal: torch.Tensor,
    positive_feature: torch.Tensor,
    encoder: TrajectoryEncoder,
    cost_model: TrajectoryCostModel,
    scales: list[float],
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    latent_scale = path.std(unbiased=False).clamp_min(1e-6)
    base_noise = torch.randn_like(path)
    base_noise[:, 0] = 0
    if mode == "intermediate":
        base_noise[:, -1] = 0
    elif mode != "all_after_start":
        raise ValueError(f"unknown mode: {mode}")

    reports: dict[str, dict[str, float]] = {}
    all_costs: list[torch.Tensor] = []
    for scale in scales:
        noisy_path = path + (scale * latent_scale) * base_noise
        feature = encoder(noisy_path, goal)
        cost = cost_model(feature)
        cosine = torch.nn.functional.cosine_similarity(feature, positive_feature, dim=-1)
        distance = (feature - positive_feature).norm(dim=-1)
        reports[f"{scale:g}"] = {
            **quantiles(cost),
            "cosine_to_expert": float(cosine.mean()),
            "l2_to_expert": float(distance.mean()),
            "feature_std": float(feature.std(unbiased=False)),
        }
        all_costs.append(cost)

    ordering: dict[str, float] = {}
    for left, right, left_cost, right_cost in zip(
        scales[:-1], scales[1:], all_costs[:-1], all_costs[1:]
    ):
        ordering[f"{left:g}<{right:g}"] = float((left_cost < right_cost).float().mean())
    return reports, ordering


def effective_rank(features: torch.Tensor) -> float:
    """Participation-ratio-like rank of the centered feature covariance."""
    centered = features - features.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    variance = singular_values.square()
    probabilities = variance / variance.sum().clamp_min(1e-12)
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    return float(entropy.exp())


def pca_projection(
    features: torch.Tensor, components: int = 3
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project features onto PCA axes and measure residual outside PC1/PC2."""
    centered = features - features.mean(dim=0, keepdim=True)
    _, singular_values, vectors = torch.pca_lowrank(
        centered,
        q=min(max(8, components), centered.size(0) - 1, centered.size(1)),
    )
    components = min(components, vectors.size(1))
    coordinates = centered @ vectors[:, :components]
    explained = singular_values.square() / centered.square().sum().clamp_min(1e-12)
    pc12 = vectors[:, : min(2, vectors.size(1))]
    residual = centered - (centered @ pc12) @ pc12.T
    return coordinates, explained, residual.norm(dim=-1)


def summarize_feature_geometry(
    features_by_scale: dict[str, torch.Tensor], scales: list[float]
) -> dict[str, object]:
    """Measure continuous versus binary-like structure of one noise mode."""
    all_features = torch.cat([features_by_scale[f"{scale:g}"] for scale in scales])
    _, explained, _ = pca_projection(all_features)
    centroids = {
        f"{scale:g}": features_by_scale[f"{scale:g}"].mean(dim=0)
        for scale in scales
    }
    within_rms = {
        key: float((features_by_scale[key] - centroids[key]).norm(dim=-1).mean())
        for key in centroids
    }
    adjacent_distances: dict[str, float] = {}
    separation_ratios: dict[str, float] = {}
    for left, right in zip(scales[:-1], scales[1:]):
        left_key, right_key = f"{left:g}", f"{right:g}"
        label = f"{left_key}->{right_key}"
        distance = float((centroids[left_key] - centroids[right_key]).norm())
        adjacent_distances[label] = distance
        separation_ratios[label] = distance / max(
            0.5 * (within_rms[left_key] + within_rms[right_key]), 1e-12
        )

    step_values = list(adjacent_distances.values())
    return {
        "effective_rank": effective_rank(all_features),
        "pca_explained_variance": {
            "pc1": float(explained[0]),
            "pc2": float(explained[1]) if explained.numel() > 1 else 0.0,
            "pc1_plus_pc2": float(explained[:2].sum()),
        },
        "within_group_rms": within_rms,
        "adjacent_centroid_l2": adjacent_distances,
        "adjacent_centroid_to_within_ratio": separation_ratios,
        # A very large first step and near-zero later steps is evidence for
        # a binary expert-vs-negative geometry.
        "first_step_fraction_of_total": (
            step_values[0] / max(sum(step_values), 1e-12)
        ),
    }


def save_feature_artifacts(
    *,
    output_path: Path,
    features: dict[str, dict[str, torch.Tensor]],
    costs: dict[str, dict[str, torch.Tensor]],
    scales: list[float],
    artifact_suffix: str = "",
    save_umap: bool = True,
    umap_neighbors: int = 30,
    umap_min_dist: float = 0.1,
    umap_random_state: int = 42,
) -> None:
    """Write PCA tables plus static and interactive feature-distribution plots."""
    rows: list[dict[str, object]] = []
    global_features: list[torch.Tensor] = []
    labels: list[tuple[str, float, int]] = []
    for mode, per_scale in features.items():
        for scale in scales:
            key = f"{scale:g}"
            feature = per_scale[key]
            global_features.append(feature)
            labels.extend((mode, scale, index) for index in range(feature.size(0)))

    joined = torch.cat(global_features)
    coordinates, _, residual_norm = pca_projection(joined, components=3)
    csv_path = output_path.with_name(
        output_path.stem + artifact_suffix + "_features_pca.csv"
    )
    offset = 0
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "mode",
                "noise_scale",
                "pc1",
                "pc2",
                "pc3",
                "residual_norm_after_pc1_pc2",
                "cost",
            ],
        )
        writer.writeheader()
        for mode, scale, index in labels:
            key = f"{scale:g}"
            cost = costs[mode][key][index]
            row = {
                "mode": mode,
                "noise_scale": scale,
                "pc1": float(coordinates[offset, 0]),
                "pc2": float(coordinates[offset, 1]),
                "pc3": float(coordinates[offset, 2]) if coordinates.size(1) > 2 else 0.0,
                "residual_norm_after_pc1_pc2": float(residual_norm[offset]),
                "cost": float(cost),
            }
            rows.append(row)
            writer.writerow(row)
            offset += 1

    save_interactive_3d_artifacts(
        output_path=output_path,
        artifact_suffix=artifact_suffix,
        rows=rows,
        csv_path=csv_path,
    )
    if save_umap:
        save_umap_3d_artifacts(
            output_path=output_path,
            artifact_suffix=artifact_suffix,
            features=joined,
            rows=rows,
            neighbors=umap_neighbors,
            min_dist=umap_min_dist,
            random_state=umap_random_state,
        )

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable: saved PCA CSV only")
        return

    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
    offset = 0
    palette = plt.cm.viridis
    denom = max(len(scales) - 1, 1)
    for axis, mode in zip(axes, features):
        for scale_index, scale in enumerate(scales):
            key = f"{scale:g}"
            count = features[mode][key].size(0)
            points = coordinates[offset : offset + count]
            offset += count
            axis.scatter(
                points[:, 0].numpy(),
                points[:, 1].numpy(),
                s=7,
                alpha=0.35,
                color=palette(scale_index / denom),
                label=f"noise={scale:g}",
            )
        axis.set_title(mode)
        axis.set_xlabel("global PCA-1")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("global PCA-2")
    axes[1].legend(markerscale=2, fontsize=8)
    figure.suptitle("LTC trajectory-encoder feature distribution")
    figure.tight_layout()
    png_path = output_path.with_name(
        output_path.stem + artifact_suffix + "_features_pca.png"
    )
    figure.savefig(png_path, dpi=180)
    plt.close(figure)
    save_static_3d_artifacts(
        output_path=output_path,
        artifact_suffix=artifact_suffix,
        rows=rows,
    )
    print(f"feature_artifacts pca_csv={csv_path} pca_png={png_path}")
def save_goal_mismatch_artifacts(
    *,
    output_path: Path,
    expert_features: torch.Tensor,
    expert_costs: torch.Tensor,
    mismatch_features: torch.Tensor,
    mismatch_costs: torch.Tensor,
) -> None:
    """Compare expert paths against identical paths conditioned on wrong goals."""
    features = torch.cat((expert_features, mismatch_features))
    coordinates, _, _ = pca_projection(features, components=3)
    labels = (["expert"] * expert_features.size(0)) + (["goal_mismatch"] * mismatch_features.size(0))
    costs = torch.cat((expert_costs, mismatch_costs))
    rows = [
        {
            "condition": label,
            "pc1": float(coordinates[index, 0]),
            "pc2": float(coordinates[index, 1]),
            "pc3": float(coordinates[index, 2]) if coordinates.size(1) > 2 else 0.0,
            "cost": float(costs[index]),
        }
        for index, label in enumerate(labels)
    ]
    csv_path = output_path.with_name(output_path.stem + "_goal_mismatch_pca.csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"matplotlib unavailable: saved goal-mismatch CSV only at {csv_path}")
        return

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for label, color in (("expert", "#4C78A8"), ("goal_mismatch", "#E45756")):
        points = [row for row in rows if row["condition"] == label]
        axes[0].scatter(
            [row["pc1"] for row in points],
            [row["pc2"] for row in points],
            s=8, alpha=.4, color=color, label=label,
        )
    axes[0].set(title="Encoder features: expert vs wrong goal", xlabel="PCA-1", ylabel="PCA-2")
    axes[0].legend(markerscale=2)
    axes[1].hist(expert_costs.numpy(), bins=30, alpha=.65, label="expert", color="#4C78A8")
    axes[1].hist(mismatch_costs.numpy(), bins=30, alpha=.65, label="goal mismatch", color="#E45756")
    axes[1].set(title="Predicted cost", xlabel="Cost (lower is better)", ylabel="Count")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=.2)
    figure.tight_layout()
    png_path = output_path.with_name(output_path.stem + "_goal_mismatch.png")
    figure.savefig(png_path, dpi=180)
    plt.close(figure)
    print(f"goal_mismatch_artifacts csv={csv_path} png={png_path}")




def save_umap_3d_artifacts(
    *,
    output_path: Path,
    artifact_suffix: str,
    features: torch.Tensor,
    rows: list[dict[str, object]],
    neighbors: int,
    min_dist: float,
    random_state: int,
) -> None:
    """Project trajectory features with UMAP and save 3-D PNG/HTML artifacts.

    UMAP is intentionally fitted jointly across both noise modes and every
    corruption scale, so nearby points have a shared geometric meaning.
    """
    try:
        import umap
    except ImportError:
        print(
            "umap-learn unavailable: skipped UMAP artifacts. "
            "Install it in the lewm environment with `pip install umap-learn`."
        )
        return

    if features.size(0) != len(rows):
        raise ValueError("UMAP feature/metadata row count mismatch")
    if features.size(0) < 3:
        print("too few feature samples for UMAP: skipped UMAP artifacts")
        return

    n_neighbors = min(max(2, int(neighbors)), features.size(0) - 1)
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=float(min_dist),
        metric="euclidean",
        random_state=int(random_state),
    )
    coordinates = reducer.fit_transform(features.detach().float().cpu().numpy())

    umap_rows: list[dict[str, object]] = []
    for row, point in zip(rows, coordinates):
        umap_row = dict(row)
        umap_row.update(
            {
                "umap1": float(point[0]),
                "umap2": float(point[1]),
                "umap3": float(point[2]),
            }
        )
        umap_rows.append(umap_row)

    csv_path = output_path.with_name(
        output_path.stem + artifact_suffix + "_features_umap.csv"
    )
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(umap_rows[0]))
        writer.writeheader()
        writer.writerows(umap_rows)

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"matplotlib unavailable: saved UMAP CSV only at {csv_path}")
    else:
        figure = plt.figure(figsize=(8, 6))
        axis = figure.add_subplot(111, projection="3d")
        palette = plt.cm.viridis
        denom = max(
            len({float(row["noise_scale"]) for row in umap_rows}) - 1,
            1,
        )
        scale_order = sorted({float(row["noise_scale"]) for row in umap_rows})
        for mode, marker in (("intermediate", "o"), ("all_after_start", "^")):
            for scale_index, scale in enumerate(scale_order):
                points = [
                    row
                    for row in umap_rows
                    if row["mode"] == mode and float(row["noise_scale"]) == scale
                ]
                axis.scatter(
                    [row["umap1"] for row in points],
                    [row["umap2"] for row in points],
                    [row["umap3"] for row in points],
                    s=7,
                    alpha=0.38,
                    color=palette(scale_index / denom),
                    marker=marker,
                    label=f"{mode}, noise={scale:g}",
                )
        axis.set_xlabel("UMAP-1")
        axis.set_ylabel("UMAP-2")
        axis.set_zlabel("UMAP-3")
        axis.set_title("LTC trajectory feature geometry: 3-D UMAP")
        axis.legend(markerscale=2, fontsize=7, ncol=2)
        figure.tight_layout()
        png_path = output_path.with_name(
            output_path.stem + artifact_suffix + "_umap3d.png"
        )
        figure.savefig(png_path, dpi=180)
        plt.close(figure)
        print(f"umap_3d_png={png_path}")

    try:
        import plotly.express as px
    except ImportError:
        print("plotly unavailable: skipped interactive UMAP HTML artifact")
        return

    figure = px.scatter_3d(
        umap_rows,
        x="umap1",
        y="umap2",
        z="umap3",
        color="noise_scale",
        symbol="mode",
        color_continuous_scale="Viridis",
        hover_data={
            "noise_scale": ":.3g",
            "cost": ":.4f",
            "umap1": ":.3f",
            "umap2": ":.3f",
            "umap3": ":.3f",
        },
        title="LTC trajectory feature geometry: 3-D UMAP",
    )
    figure.update_traces(marker={"size": 3, "opacity": 0.55})
    figure.update_layout(scene={"aspectmode": "data"})
    html_path = output_path.with_name(
        output_path.stem + artifact_suffix + "_umap3d.html"
    )
    figure.write_html(html_path, include_plotlyjs=True, full_html=True)
    print(f"umap_3d_html={html_path} source_csv={csv_path}")

def save_static_3d_artifacts(
    *,
    output_path: Path,
    artifact_suffix: str,
    rows: list[dict[str, object]],
) -> None:
    """Write static 3-D PNG fallbacks when interactive Plotly is unavailable."""
    import matplotlib.pyplot as plt

    color_map = {"intermediate": "#4C78A8", "all_after_start": "#F58518"}
    marker_map = {"intermediate": "o", "all_after_start": "^"}
    for z_key, z_label, suffix in (
        ("pc3", "PC3", "_pca3d"),
        (
            "residual_norm_after_pc1_pc2",
            "Residual norm after PC1/PC2",
            "_pc12_residual3d",
        ),
    ):
        figure = plt.figure(figsize=(8, 6))
        axis = figure.add_subplot(111, projection="3d")
        for mode in ("intermediate", "all_after_start"):
            points = [row for row in rows if row["mode"] == mode]
            axis.scatter(
                [row["pc1"] for row in points],
                [row["pc2"] for row in points],
                [row[z_key] for row in points],
                s=7,
                alpha=0.35,
                color=color_map[mode],
                marker=marker_map[mode],
                label=mode,
            )
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
        axis.set_zlabel(z_label)
        axis.set_title(f"LTC feature geometry: PC1 / PC2 / {z_label}")
        axis.legend(markerscale=2)
        figure.tight_layout()
        figure.savefig(
            output_path.with_name(output_path.stem + artifact_suffix + suffix + ".png"),
            dpi=180,
        )
        plt.close(figure)



def save_interactive_3d_artifacts(
    *,
    output_path: Path,
    artifact_suffix: str,
    rows: list[dict[str, object]],
    csv_path: Path,
) -> None:
    """Create self-contained, rotatable Plotly views when Plotly is installed."""
    try:
        import plotly.express as px
    except ImportError:
        print("plotly unavailable: skipped interactive 3-D HTML artifacts")
        return

    color_map = {"intermediate": "#4C78A8", "all_after_start": "#F58518"}
    common = dict(
        color="mode",
        symbol="noise_scale",
        color_discrete_map=color_map,
        hover_data={
            "noise_scale": ":.3g",
            "cost": ":.4f",
            "pc1": ":.3f",
            "pc2": ":.3f",
            "pc3": ":.3f",
            "residual_norm_after_pc1_pc2": ":.3f",
        },
    )
    figures = {
        "_pca3d": px.scatter_3d(rows, x="pc1", y="pc2", z="pc3", title="LTC feature geometry: PC1 / PC2 / PC3", **common),
        "_pc12_residual3d": px.scatter_3d(rows, x="pc1", y="pc2", z="residual_norm_after_pc1_pc2", title="LTC feature geometry: PC1 / PC2 / residual norm", **common),
    }
    for suffix, figure in figures.items():
        figure.update_traces(marker={"size": 3, "opacity": 0.55})
        figure.update_layout(legend_title_text="mode / noise scale", scene={"aspectmode": "data"})
        html_path = output_path.with_name(output_path.stem + artifact_suffix + suffix + ".html")
        figure.write_html(html_path, include_plotlyjs=True, full_html=True)
        print(f"interactive_3d_html={html_path} source_csv={csv_path}")


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    scales = [float(value) for value in args.noise_scales.split(",")]
    if scales[0] != 0.0 or any(b <= a for a, b in zip(scales[:-1], scales[1:])):
        raise ValueError("--noise-scales must start with 0 and be strictly increasing")

    loader = make_loader(args)
    lewm = load_lewm(args.lewm_checkpoint).to(device).eval()
    lewm.requires_grad_(False)

    first = next(iter(loader))
    first_path = lewm.encode({"pixels": first["pixels"].to(device)})["emb"]
    encoder, cost_model = load_ltc(
        args.ltc_checkpoint, latent_dim=first_path.size(-1), device=device
    )
    if first_path.size(1) > encoder.max_horizon + 1:
        raise ValueError("LTC encoder max_horizon is smaller than the inspected path")

    stores = {
        mode: {f"{scale:g}": [] for scale in scales}
        for mode in ("intermediate", "all_after_start")
    }
    cosine_stores = {
        mode: {f"{scale:g}": [] for scale in scales}
        for mode in ("intermediate", "all_after_start")
    }
    distance_stores = {
        mode: {f"{scale:g}": [] for scale in scales}
        for mode in ("intermediate", "all_after_start")
    }
    feature_stores = {
        mode: {f"{scale:g}": [] for scale in scales}
        for mode in ("intermediate", "all_after_start")
    }
    # Removes each task's expert feature, isolating the representation change
    # attributable to corruption rather than start/goal identity.
    delta_feature_stores = {
        mode: {f"{scale:g}": [] for scale in scales}
        for mode in ("intermediate", "all_after_start")
    }
    mismatch_cost_store: list[torch.Tensor] = []
    mismatch_feature_store: list[torch.Tensor] = []
    mismatch_cosine_store: list[torch.Tensor] = []
    mismatch_distance_store: list[torch.Tensor] = []
    expert_cost_store: list[torch.Tensor] = []
    expert_feature_store: list[torch.Tensor] = []

    for batch_index, batch in enumerate(loader):
        path = lewm.encode({"pixels": batch["pixels"].to(device)})["emb"]
        goal = path[:, -1]
        positive_feature = encoder(path, goal)
        latent_scale = path.std(unbiased=False).clamp_min(1e-6)

        for mode in stores:
            base_noise = torch.randn_like(path)
            base_noise[:, 0] = 0
            if mode == "intermediate":
                base_noise[:, -1] = 0
            for scale in scales:
                feature = encoder(path + scale * latent_scale * base_noise, goal)
                costs = cost_model(feature)
                key = f"{scale:g}"
                stores[mode][key].append(costs.cpu())
                cosine_stores[mode][key].append(
                    torch.nn.functional.cosine_similarity(feature, positive_feature, dim=-1).cpu()
                )
                distance_stores[mode][key].append((feature - positive_feature).norm(dim=-1).cpu())
                feature_stores[mode][key].append(feature.cpu())
                delta_feature_stores[mode][key].append(
                    (feature - positive_feature).cpu()
                )

        expert_cost = cost_model(positive_feature)
        mismatch_goal = goal.roll(shifts=1, dims=0)
        mismatch_feature = encoder(path, mismatch_goal)
        mismatch_cost = cost_model(mismatch_feature)
        expert_cost_store.append(expert_cost.cpu())
        expert_feature_store.append(positive_feature.cpu())
        mismatch_cost_store.append(mismatch_cost.cpu())
        mismatch_feature_store.append(mismatch_feature.cpu())
        mismatch_cosine_store.append(
            torch.nn.functional.cosine_similarity(
                mismatch_feature, positive_feature, dim=-1
            ).cpu()
        )
        mismatch_distance_store.append(
            (mismatch_feature - positive_feature).norm(dim=-1).cpu()
        )
        if batch_index + 1 >= args.num_batches:
            break

    output: dict[str, object] = {
        "checkpoint": {
            "ltc_checkpoint": args.ltc_checkpoint,
            "encoder_input": "path_only",
        },
        "dataset": args.dataset,
        "path_horizon": args.horizon,
        "action_block": args.action_block,
        "noise_scales": scales,
        "modes": {},
    }
    print(f"samples_per_scale={sum(x.numel() for x in next(iter(stores['intermediate'].values())))}")
    for mode in stores:
        costs_by_scale = {
            key: torch.cat(values) for key, values in stores[mode].items()
        }
        mode_summary: dict[str, object] = {"scales": {}, "adjacent_order_rate": {}}
        print(f"\n[{mode}]")
        print("scale    mean+/-std         p05 / p50 / p95      cosine_to_expert  l2_to_expert")
        for scale in scales:
            key = f"{scale:g}"
            stats = quantiles(costs_by_scale[key])
            cosine = torch.cat(cosine_stores[mode][key]).mean()
            distance = torch.cat(distance_stores[mode][key]).mean()
            stats["cosine_to_expert"] = float(cosine)
            stats["l2_to_expert"] = float(distance)
            mode_summary["scales"][key] = stats
            print(
                f"{key:>5}  {stats['mean']:>8.4f}+/-{stats['std']:<7.4f} "
                f"{stats['p05']:>8.4f} / {stats['p50']:>8.4f} / {stats['p95']:>8.4f} "
                f"{float(cosine):>8.4f}          {float(distance):>8.4f}"
            )
        for left, right in zip(scales[:-1], scales[1:]):
            left_key, right_key = f"{left:g}", f"{right:g}"
            rate = (costs_by_scale[left_key] < costs_by_scale[right_key]).float().mean()
            label = f"{left_key}<{right_key}"
            mode_summary["adjacent_order_rate"][label] = float(rate)
            print(f"order_rate cost({left_key}) < cost({right_key}): {float(rate):.3f}")
        features_by_scale = {
            key: torch.cat(values) for key, values in feature_stores[mode].items()
        }
        geometry = summarize_feature_geometry(features_by_scale, scales)
        delta_features_by_scale = {
            key: torch.cat(values)
            for key, values in delta_feature_stores[mode].items()
        }
        centered_geometry = summarize_feature_geometry(delta_features_by_scale, scales)
        mode_summary["feature_geometry"] = geometry
        mode_summary["expert_centered_feature_geometry"] = centered_geometry
        for title, result in (
            ("raw_feature_geometry", geometry),
            ("expert_centered_feature_geometry", centered_geometry),
        ):
            print(
                f"{title} "
                f"effective_rank={result['effective_rank']:.2f} "
                f"pca_pc1={result['pca_explained_variance']['pc1']:.3f} "
                f"pca_pc1_plus_pc2={result['pca_explained_variance']['pc1_plus_pc2']:.3f} "
                f"first_step_fraction={result['first_step_fraction_of_total']:.3f}"
            )
            for label, value in result["adjacent_centroid_l2"].items():
                ratio = result["adjacent_centroid_to_within_ratio"][label]
                print(
                    f"{title}_centroid_step {label}: "
                    f"l2={value:.4f} centroid_to_within_ratio={ratio:.3f}"
                )
        output["modes"][mode] = mode_summary

    expert_costs = torch.cat(expert_cost_store)
    expert_features = torch.cat(expert_feature_store)
    mismatch_costs = torch.cat(mismatch_cost_store)
    mismatch_features = torch.cat(mismatch_feature_store)
    mismatch_summary = {
        **quantiles(mismatch_costs),
        "cost_minus_expert_mean": float((mismatch_costs - expert_costs).mean()),
        "cost_greater_than_expert_rate": float((mismatch_costs > expert_costs).float().mean()),
        "cosine_to_expert": float(torch.cat(mismatch_cosine_store).mean()),
        "l2_to_expert": float(torch.cat(mismatch_distance_store).mean()),
        "feature_std": float(mismatch_features.std(unbiased=False)),
    }
    output["goal_mismatch"] = mismatch_summary
    print("\n[goal_mismatch]")
    print(
        "cost="
        f"{mismatch_summary['mean']:.4f}+/-{mismatch_summary['std']:.4f} "
        f"delta={mismatch_summary['cost_minus_expert_mean']:.4f} "
        f"order_rate={mismatch_summary['cost_greater_than_expert_rate']:.3f} "
        f"cosine={mismatch_summary['cosine_to_expert']:.4f} "
        f"l2={mismatch_summary['l2_to_expert']:.4f}"
    )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2))
        features = {
            mode: {key: torch.cat(values) for key, values in per_scale.items()}
            for mode, per_scale in feature_stores.items()
        }
        costs = {
            mode: {key: torch.cat(values) for key, values in per_scale.items()}
            for mode, per_scale in stores.items()
        }
        save_feature_artifacts(
            output_path=output_path,
            features=features,
            costs=costs,
            scales=scales,
            save_umap=not args.skip_umap,
            umap_neighbors=args.umap_neighbors,
            umap_min_dist=args.umap_min_dist,
            umap_random_state=args.umap_random_state,
        )
        delta_features = {
            mode: {key: torch.cat(values) for key, values in per_scale.items()}
            for mode, per_scale in delta_feature_stores.items()
        }
        save_feature_artifacts(
            output_path=output_path,
            features=delta_features,
            costs=costs,
            scales=scales,
            artifact_suffix="_expert_centered",
            save_umap=not args.skip_umap,
            umap_neighbors=args.umap_neighbors,
            umap_min_dist=args.umap_min_dist,
            umap_random_state=args.umap_random_state,
        )
        save_goal_mismatch_artifacts(
            output_path=output_path,
            expert_features=expert_features,
            expert_costs=expert_costs,
            mismatch_features=mismatch_features,
            mismatch_costs=mismatch_costs,
        )
        print(f"summary_written={args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lewm-checkpoint", required=True)
    parser.add_argument("--ltc-checkpoint", required=True)
    parser.add_argument("--dataset", default="pusht_expert_train")
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-batches", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--noise-scales", default="0,0.02,0.05,0.1,0.2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--umap-neighbors",
        type=int,
        default=30,
        help="Neighborhood size for the joint 3-D UMAP projection.",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.1,
        help="Minimum distance for the joint 3-D UMAP projection.",
    )
    parser.add_argument(
        "--umap-random-state",
        type=int,
        default=42,
        help="Fixed seed for reproducible UMAP coordinates.",
    )
    parser.add_argument(
        "--skip-umap",
        action="store_true",
        help="Skip optional UMAP artifacts while retaining PCA artifacts.",
    )
    parser.add_argument("--output", default=None)
    run(parser.parse_args())
