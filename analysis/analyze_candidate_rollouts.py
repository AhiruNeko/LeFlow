"""Same-noise, same-task comparison of experience FIFO candidate schedules."""
from __future__ import annotations
import argparse, csv, json, os, sys
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
from sklearn.preprocessing import StandardScaler
import stable_worldmodel as swm
from latent_planner import LatentPlannerRuntime, load_ltc_components_for_evaluation
from stable_worldmodel.envs.pusht.env import PushT
from utils import get_img_preprocessor

METRICS = {
    "ltc_cost": "LTC cost",
    "lewm_rollout_ltc_cost": "LTC cost of LeWM rollout path",
    "lewm_goal_distance": "LeWM terminal goal MSE",
    "lewm_path_distance": "LeWM path-consistency MSE",
    "lewm_nominal_segment_distance": (
        "Mean independent LeWM segment defect (nominal planner subgoals)"
    ),
    "expert_path_distance": "Expert latent trajectory MSE",
    "simulator_distance": "PushT simulator terminal state distance",
}

def args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--planner-checkpoint", type=Path, required=True)
    p.add_argument("--dataset-name", default="pusht_expert_train")
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--episode-index", type=int, default=0)
    p.add_argument("--start-step", type=int, default=0)
    p.add_argument("--goal-offset-steps", type=int, default=50)
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--action-block", type=int, default=5)
    p.add_argument("--flow-steps", type=int, default=16)
    p.add_argument("--history-size", type=int, default=3)
    p.add_argument("--max-paths", type=int, default=64)
    p.add_argument("--schedules", default="64x1,32x2,16x4,8x8,4x16")
    p.add_argument("--noise-seed", type=int, default=20260928)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output-dir", type=Path,
                   default=Path("analysis/candidate_rollout_comparison"))
    return p.parse_args()

def parse_schedules(text, total):
    out = []
    for value in text.split(","):
        name = value.strip().lower()
        try:
            n, r = (int(x) for x in name.split("x"))
        except ValueError as exc:
            raise ValueError(f"Invalid schedule {value!r}; use 32x2.") from exc
        if n < 1 or r < 1 or n * r != total:
            raise ValueError(f"{name!r} must generate exactly {total} paths.")
        out.append((name, n, r))
    return out

def np_value(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

def image_latent(runtime, image, prep, device):
    image = torch.as_tensor(image)
    if image.ndim == 3 and image.shape[-1] in (1, 3):
        image = image.permute(2, 0, 1)
    if image.ndim != 3:
        raise ValueError(f"Invalid image shape {tuple(image.shape)}")
    pixels = prep({"pixels": image[None]})["pixels"][None].to(device)
    return runtime.lewm.encode({"pixels": pixels})["emb"][:, -1]

@torch.no_grad()
def sample_fixed_noise(runtime, z0, zg, noise, flow_steps, feature=None, cost=None):
    if (feature is None) != (cost is None):
        raise ValueError("feature and cost must be supplied together")
    b, n, _, d = noise.shape
    x = noise.reshape(b * n, noise.size(2), d).clone()
    start = z0[:, None].expand(-1, n, -1).reshape(b * n, d)
    goal = zg[:, None].expand(-1, n, -1).reshape(b * n, d)
    feat = None if feature is None else feature.repeat_interleave(n, 0)
    costs = None if cost is None else cost.repeat_interleave(n, 0)
    dt = 1.0 / max(flow_steps, 1)
    for i in range(flow_steps):
        t = torch.full((b * n,), i * dt, device=x.device, dtype=x.dtype)
        x += runtime.flow(x, t, start, goal, path_features=feat, path_costs=costs) * dt
    return torch.cat([start[:, None], x, goal[:, None]], 1).reshape(
        b, n, x.size(1) + 2, d
    )

@torch.no_grad()
def generate(runtime, encoder, scorer, z0, zg, noise, n, rounds, cap, steps):
    memory_feature = memory_cost = None
    paths_all, costs_all = [], []
    cursor = 0
    for _ in range(rounds):
        paths = sample_fixed_noise(
            runtime, z0, zg, noise[:, cursor:cursor+n], steps,
            memory_feature, memory_cost,
        )
        flat = paths.flatten(0, 1)
        goals = zg[:, None].expand(-1, n, -1).reshape(-1, zg.size(-1))
        feature = encoder(flat, goals).reshape(1, n, -1)
        costs = scorer(feature.flatten(0, 1)).reshape(1, n)
        memory_feature = feature if memory_feature is None else torch.cat(
            [memory_feature, feature], 1
        )
        memory_cost = costs if memory_cost is None else torch.cat([memory_cost, costs], 1)
        memory_feature, memory_cost = memory_feature[:, -cap:], memory_cost[:, -cap:]
        paths_all.append(paths)
        costs_all.append(costs)
        cursor += n
    return torch.cat(paths_all, 1), torch.cat(costs_all, 1)

def simulation_options(chunk):
    opt = {
        "state": np_value(chunk["state"][0]),
        # PushT expert datasets store the demonstrated state sequence rather
        # than an explicit goal_state column. The final state of this task
        # segment is its selected goal, matching chunk["pixels"][-1] used by
        # the latent planner below.
        "goal_state": np_value(chunk["state"][-1]),
    }
    variation = [k for k in chunk if k.startswith("variation.")]
    if variation:
        opt["variation"] = [k.removeprefix("variation.") for k in variation]
        opt["variation_values"] = {
            k.removeprefix("variation."): np_value(chunk[k][0]) for k in variation
        }
    seed = None if "seed" not in chunk else int(np_value(chunk["seed"])[0])
    return seed, opt

def simulator_distances(actions, scaler, block, seed, options):
    actions = actions.detach().cpu().numpy()[0]
    n, h, packed = actions.shape
    action_dim = packed // block
    if action_dim * block != packed or action_dim != 2:
        raise ValueError(f"Expected PushT packed actions [...,{block}*2], got {packed}.")
    output = np.empty(n)
    goal = np.asarray(options["goal_state"])
    for i, candidate in enumerate(actions):
        env = PushT(resolution=224, render_mode="rgb_array")
        env.reset(seed=seed, options=options)
        raw = scaler.inverse_transform(candidate.reshape(h * block, action_dim))
        for action in raw:
            env.step(action.astype(np.float32, copy=False))
        _, output[i] = env.eval_state(goal, env._get_obs())
        env.close()
    return output

@torch.no_grad()
def nominal_segment_defects(runtime, paths, actions, history_size):
    """Measure each planned transition without propagating LeWM predictions.

    For transition t, the LeWM context uses the original planner states and
    actions ending at p_t.  This is a multiple-shooting-style local defect,
    unlike rollout_paths(), which feeds LeWM's own predicted state into the
    next transition.
    """
    batch, candidates, horizon, latent_dim = paths.shape
    action_dim = actions.size(-1)
    defects = []
    for t in range(horizon - 1):
        start = max(0, t - history_size + 1)
        context = paths[:, :, start : t + 1].reshape(
            batch * candidates, t - start + 1, latent_dim
        )
        action_context = actions[:, :, start : t + 1].reshape(
            batch * candidates, t - start + 1, action_dim
        )
        predicted = runtime.lewm.predict(
            context, runtime.lewm.action_encoder(action_context)
        )[:, -1]
        target = paths[:, :, t + 1].reshape(batch * candidates, latent_dim)
        defects.append((predicted - target).square().mean(dim=-1).reshape(
            batch, candidates
        ))
    return torch.stack(defects, dim=-1)

@torch.no_grad()
def score_paths(
    runtime, encoder, scorer, paths, costs, z0, zg, expert_path,
    hist, scaler, block, seed, options,
):
    actions = runtime.decode_actions(paths)
    rollout = runtime.rollout_paths(z0, actions, hist)

    # Re-score the trajectory that LeWM predicts will actually result after
    # inverse-dynamics decoding. This is distinct from the planner-path cost.
    batch, candidates, _, latent_dim = rollout.shape
    rollout_goals = zg[:, None].expand(-1, candidates, -1).reshape(-1, latent_dim)
    rollout_features = encoder(rollout.flatten(0, 1), rollout_goals)
    rollout_costs = scorer(rollout_features).reshape(batch, candidates)

    terminal = (rollout[:, :, -1] - zg[:, None]).square().mean(-1)
    path = (rollout[:, :, 1:] - paths[:, :, 1:]).square().mean(-1).mean(-1)
    nominal_defects = nominal_segment_defects(runtime, paths, actions, hist)
    nominal_segment_mean = nominal_defects.mean(dim=-1)
    # Candidate and expert trajectories share z_start and z_goal. This metric
    # compares their full aligned latent paths, including those fixed endpoints.
    expert = (paths - expert_path[:, None]).square().mean(-1).mean(-1)
    return {
        "ltc_cost": costs[0].cpu().numpy(),
        "lewm_rollout_ltc_cost": rollout_costs[0].cpu().numpy(),
        "lewm_goal_distance": terminal[0].cpu().numpy(),
        "lewm_path_distance": path[0].cpu().numpy(),
        "lewm_nominal_segment_distance": nominal_segment_mean[0].cpu().numpy(),
        "expert_path_distance": expert[0].cpu().numpy(),
        "simulator_distance": simulator_distances(actions, scaler, block, seed, options),
    }

def mark_minimum(ax, x, y, label, color):
    """Mark one schedule's minimum with explicit candidate/value coordinates."""
    index = int(np.argmin(y))
    ax.scatter(
        x[index], y[index], marker="*", s=80, color=color,
        edgecolors="black", linewidths=.5, zorder=4,
    )
    ax.annotate(
        f"{label}: ({x[index]}, {y[index]:.4g})",
        xy=(x[index], y[index]),
        xytext=(5, 5),
        textcoords="offset points",
        color=color,
        fontsize=7,
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": color, "alpha": .72},
    )


def save_and_plot(records, output):
    rows = []
    for schedule, values in records.items():
        for i in range(len(values["ltc_cost"])):
            rows.append({"schedule": schedule, "candidate_index": i + 1,
                         **{key: float(values[key][i]) for key in METRICS}})
    with (output / "candidate_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(output / "candidate_metrics.npz", **{
        f"{schedule}_{metric}": values[metric]
        for schedule, values in records.items() for metric in METRICS
    })
    x = np.arange(1, len(rows) // len(records) + 1)
    columns = 2
    panel_rows = int(np.ceil(len(METRICS) / columns))
    fig, axes = plt.subplots(
        panel_rows, columns, figsize=(16, 5 * panel_rows), constrained_layout=True
    )
    for ax, (metric, title) in zip(axes.flat, METRICS.items()):
        for schedule, values in records.items():
            y = values[metric]
            line = ax.plot(x, y, label=schedule, lw=1.2)[0]
            mark_minimum(ax, x, y, schedule, line.get_color())
        ax.set(title=title, xlabel="Candidate generation index",
               ylabel="Lower is better")
        ax.grid(alpha=.25); ax.legend(title="Candidates x rounds")
    for ax in axes.flat[len(METRICS):]:
        ax.set_visible(False)
    fig.suptitle("Same task and fixed initial diffusion noise", fontsize=14)
    fig.savefig(output / "candidate_metrics_comparison.png", dpi=180)
    plt.close(fig)
    for metric, title in METRICS.items():
        fig, ax = plt.subplots(figsize=(12, 5))
        for schedule, values in records.items():
            y = values[metric]
            line = ax.plot(x, y, label=schedule, lw=1.5)[0]
            mark_minimum(ax, x, y, schedule, line.get_color())
        ax.set(title=title, xlabel="Candidate generation index",
               ylabel="Lower is better")
        ax.grid(alpha=.25); ax.legend(title="Candidates x rounds")
        fig.tight_layout(); fig.savefig(output / f"{metric}.png", dpi=180)
        plt.close(fig)

def main():
    cfg = args()
    if cfg.horizon * cfg.action_block != cfg.goal_offset_steps:
        raise ValueError("goal-offset-steps must equal horizon * action-block.")
    schedules = parse_schedules(cfg.schedules, cfg.max_paths)
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
        raise ValueError("LTC and flow path_feature_dim differ.")
    if runtime.action_block != cfg.action_block:
        raise ValueError("action-block differs from checkpoint.")
    dataset = swm.data.HDF5Dataset(
        cfg.dataset_name, frameskip=1, num_steps=1, cache_dir=cfg.cache_dir
    )
    if not 0 <= cfg.episode_index < len(dataset.lengths):
        raise ValueError("episode-index is outside dataset.")
    if cfg.start_step < 0 or cfg.start_step + cfg.goal_offset_steps >= dataset.lengths[cfg.episode_index]:
        raise ValueError("Selected task exceeds its episode.")
    chunk = dataset.load_chunk(
        np.asarray([cfg.episode_index]), np.asarray([cfg.start_step]),
        np.asarray([cfg.start_step + cfg.goal_offset_steps]),
    )[0]
    missing = {"pixels", "state", "action"} - set(chunk)
    if missing:
        raise KeyError(f"Dataset task missing {sorted(missing)}")
    prep = get_img_preprocessor("pixels", "pixels", cfg.img_size)
    z0 = image_latent(runtime, chunk["pixels"][0], prep, device)
    zg = image_latent(runtime, chunk["pixels"][-1], prep, device)
    # HDF5Dataset.load_chunk uses an exclusive end index, so a 50-step task
    # contains frames 0..49. Uniformly resample that available expert segment
    # to the planner's H+1 latent states; this preserves exact shared endpoints.
    expert_indices = np.rint(
        np.linspace(0, len(chunk["pixels"]) - 1, cfg.horizon + 1)
    ).astype(np.int64)
    expert_path = torch.cat(
        [image_latent(runtime, chunk["pixels"][index], prep, device)[:, None]
         for index in expert_indices],
        dim=1,
    )
    generator = torch.Generator(device=device).manual_seed(cfg.noise_seed)
    fixed_noise = torch.randn(
        1, cfg.max_paths, cfg.horizon - 1, runtime.flow.latent_dim,
        device=device, generator=generator,
    )
    action_data = np.asarray(dataset.get_col_data("action"), dtype=np.float64).reshape(-1, 2)
    scaler = StandardScaler().fit(action_data[np.isfinite(action_data).all(1)])
    seed, options = simulation_options(chunk)
    records = {}
    for name, n, rounds in schedules:
        print(f"schedule={name} generation_start", flush=True)
        paths, costs = generate(
            runtime, encoder, scorer, z0, zg, fixed_noise, n, rounds,
            cfg.max_paths, cfg.flow_steps,
        )
        records[name] = score_paths(
            runtime, encoder, scorer, paths, costs, z0, zg, expert_path,
            cfg.history_size, scaler, cfg.action_block, seed, options,
        )
        print("schedule=" + name + " " + " ".join(
            f"{key}_min={values[key].min():.5f}"
            for key, values in [(key, records[name]) for key in METRICS]
        ), flush=True)
    output = cfg.output_dir / (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + f"_ep{cfg.episode_index}_step{cfg.start_step}"
    )
    output.mkdir(parents=True, exist_ok=False)
    save_and_plot(records, output)
    meta = vars(cfg) | {
        "planner_checkpoint": str(cfg.planner_checkpoint),
        "output_dir": str(cfg.output_dir),
        "cache_dir": None if cfg.cache_dir is None else str(cfg.cache_dir),
        "fixed_noise_shape": list(fixed_noise.shape),
        "schedules": [x[0] for x in schedules],
        "simulation_seed": seed,
        "expert_frame_offsets": expert_indices.tolist(),
        "simulation_goal_state": np_value(options["goal_state"]).tolist(),
    }
    with (output / "metadata.json").open("w") as f:
        json.dump(meta, f, indent=2)
    print(f"analysis_complete output_dir={output}", flush=True)

if __name__ == "__main__":
    main()
