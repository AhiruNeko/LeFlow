"""Compare planner-path, real-execution, and LeWM-rollout trajectory rankings.

LeWM has no latent-to-image decoder. Raw planner and LeWM rollout latents are
shown through nearest-frame projection onto real images from the source episode.
The simulator group uses inverse-dynamics actions executed in real PushT.
"""
from __future__ import annotations

import argparse
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

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from PIL import Image, ImageDraw
import stable_worldmodel as swm

from analyze_candidate_rollouts import generate, image_latent, simulation_options
from latent_planner import LatentPlannerRuntime, load_ltc_components_for_evaluation
from stable_worldmodel.envs.pusht.env import PushT
from utils import get_img_preprocessor


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--experience-max-size", type=int, default=64)
    parser.add_argument("--noise-seed", type=int, default=20260928)
    parser.add_argument(
        "--num-rendered-paths",
        type=int,
        default=4,
        help="Render this many best candidates for each of the three ranking modes.",
    )
    parser.add_argument(
        "--retrieval-stride",
        type=int,
        default=1,
        help="Use every Nth source-episode frame in the latent nearest-neighbor bank.",
    )
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--thumb-size", type=int, default=160)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/trajectory_rendering"),
    )
    return parser.parse_args()


def to_rgb(image: np.ndarray | torch.Tensor) -> Image.Image:
    array = np.asarray(image.detach().cpu() if isinstance(image, torch.Tensor) else image)
    if array.ndim != 3:
        raise ValueError(f"Expected image [H,W,C] or [C,H,W], got {array.shape}")
    if array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.moveaxis(array, 0, -1)
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] != 3:
        raise ValueError(f"Expected RGB image, got {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        scale = 255.0 if float(array.max(initial=0.0)) <= 1.0 else 1.0
        array = array * scale
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB")


def put_label(image: Image.Image, text: str, color: str) -> Image.Image:
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, canvas.width - 1, canvas.height - 1), outline=color, width=5)
    draw.rectangle((0, 0, canvas.width, 18), fill=(255, 255, 255))
    draw.text((4, 3), text, fill=color)
    return canvas


def frame_strip(frames: list[Image.Image], title: str, thumb_size: int) -> Image.Image:
    cols = min(6, len(frames))
    rows = int(np.ceil(len(frames) / cols))
    label_height = 22
    canvas = Image.new(
        "RGB",
        (cols * thumb_size, 30 + rows * (thumb_size + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 6), title, fill="black")
    for index, frame in enumerate(frames):
        thumb = frame.resize((thumb_size, thumb_size), Image.Resampling.BILINEAR)
        row, col = divmod(index, cols)
        x = col * thumb_size
        y = 30 + row * (thumb_size + label_height)
        canvas.paste(thumb, (x, y))
        if index == 0:
            color, name = "forestgreen", "start"
        elif index == len(frames) - 1:
            color, name = "firebrick", "goal"
        else:
            color, name = "gray", "intermediate"
        ImageDraw.Draw(canvas).rectangle(
            (x, y, x + thumb_size - 1, y + thumb_size - 1), outline=color, width=4
        )
        ImageDraw.Draw(canvas).text((x + 3, y + thumb_size + 3), f"t={index} {name}", fill=color)
    return canvas


def foreground_rgba(frame: Image.Image, alpha: int) -> Image.Image:
    """Keep only non-white PushT foreground pixels at the requested alpha."""
    rgb = np.asarray(frame.convert("RGB"))
    foreground = np.max(np.abs(rgb.astype(np.int16) - 255), axis=-1) > 12
    rgba = np.dstack((rgb, np.where(foreground, alpha, 0).astype(np.uint8)))
    return Image.fromarray(rgba, "RGBA")


def path_overlay(frames: list[Image.Image], title: str) -> Image.Image:
    if not frames:
        raise ValueError("Cannot overlay an empty trajectory")
    width, height = frames[0].size
    canvas = Image.new("RGBA", (width, height + 24), (255, 255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 4), title, fill="black")

    # Suppress white backgrounds before compositing. This makes every
    # intermediate object pose visible instead of repeatedly whitening the
    # previous image. Start and goal are opaque anchors.
    for index, frame in enumerate(frames):
        alpha = 255 if index in (0, len(frames) - 1) else 112
        canvas.alpha_composite(foreground_rgba(frame, alpha), (0, 24))

    draw.rectangle((1, 25, width - 2, height + 22), outline="forestgreen", width=5)
    draw.rectangle((7, 31, width - 8, height + 16), outline="firebrick", width=5)
    draw.text((8, height + 4), "green outer = start; red inner = goal", fill="black")
    return canvas.convert("RGB")


def stack_vertical(images: list[Image.Image], padding: int = 12) -> Image.Image:
    width = max(image.width for image in images)
    height = sum(image.height for image in images) + padding * (len(images) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    for image in images:
        canvas.paste(image, ((width - image.width) // 2, y))
        y += image.height + padding
    return canvas


def expert_frames(chunk: dict, horizon: int) -> list[Image.Image]:
    indices = np.rint(np.linspace(0, len(chunk["pixels"]) - 1, horizon + 1)).astype(np.int64)
    return [to_rgb(chunk["pixels"][index]) for index in indices]


@torch.no_grad()
def episode_latent_bank(
    runtime: LatentPlannerRuntime,
    images: np.ndarray,
    prep,
    device: torch.device,
    stride: int,
) -> tuple[np.ndarray, torch.Tensor]:
    if stride < 1:
        raise ValueError("retrieval-stride must be positive")
    indices = np.arange(0, len(images), stride, dtype=np.int64)
    if indices[-1] != len(images) - 1:
        indices = np.append(indices, len(images) - 1)
    latents = torch.cat(
        [image_latent(runtime, images[index], prep, device) for index in indices],
        dim=0,
    )
    return indices, latents


@torch.no_grad()
def nearest_episode_frames(
    path: torch.Tensor,
    bank_indices: np.ndarray,
    bank_latents: torch.Tensor,
    episode_images: np.ndarray,
) -> tuple[list[Image.Image], np.ndarray, np.ndarray]:
    distances = torch.cdist(path[None], bank_latents[None]).squeeze(0)
    nearest_bank = distances.argmin(dim=1)
    nearest_indices = bank_indices[nearest_bank.detach().cpu().numpy()]
    nearest_distances = distances[
        torch.arange(path.size(0), device=path.device), nearest_bank
    ].detach().cpu().numpy()
    frames = [to_rgb(episode_images[index]) for index in nearest_indices]
    return frames, nearest_indices, nearest_distances


def simulator_rollout(
    actions: torch.Tensor,
    scaler: StandardScaler,
    action_block: int,
    seed: int | None,
    options: dict,
) -> tuple[list[Image.Image], float]:
    """Execute a decoded packed action path in real PushT and render each block."""
    packed = actions.detach().cpu().numpy()
    if packed.ndim != 2:
        raise ValueError(f"Expected [H, packed_action_dim], got {packed.shape}")
    action_dim = packed.shape[-1] // action_block
    if action_dim != 2 or action_dim * action_block != packed.shape[-1]:
        raise ValueError("PushT requires packed action dimension action_block * 2.")

    env = PushT(resolution=224, render_mode="rgb_array")
    try:
        observation, _ = env.reset(seed=seed, options=options)
        frames = [to_rgb(env.render())]
        raw_actions = scaler.inverse_transform(packed.reshape(-1, action_dim))
        for step in range(packed.shape[0]):
            for action in raw_actions[step * action_block : (step + 1) * action_block]:
                observation, _, _, _, _ = env.step(action.astype(np.float32, copy=False))
            frames.append(to_rgb(env.render()))
        _, terminal_distance = env.eval_state(
            np.asarray(options["goal_state"]), observation["state"]
        )
        return frames, float(terminal_distance)
    finally:
        env.close()


@torch.no_grad()
def main() -> None:
    cfg = parse_args()
    if cfg.horizon * cfg.action_block != cfg.goal_offset_steps:
        raise ValueError("goal-offset-steps must equal horizon * action-block.")
    if cfg.candidates < 1 or cfg.rounds < 1:
        raise ValueError("candidates and rounds must be positive.")

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    payload = torch.load(cfg.planner_checkpoint, map_location="cpu", weights_only=False)
    runtime = LatentPlannerRuntime.from_checkpoint(cfg.planner_checkpoint, device).eval()
    ltc = payload.get("experience", {}).get("ltc")
    if ltc is None:
        raise ValueError("Planner checkpoint has no bundled experience.ltc payload.")
    encoder, scorer = load_ltc_components_for_evaluation(
        ltc, latent_dim=runtime.flow.latent_dim, device=device
    )
    if runtime.action_block != cfg.action_block:
        raise ValueError("action-block does not match the planner checkpoint.")

    dataset = swm.data.HDF5Dataset(
        cfg.dataset_name, frameskip=1, num_steps=1, cache_dir=cfg.cache_dir
    )
    if not 0 <= cfg.episode_index < len(dataset.lengths):
        raise ValueError("episode-index is outside the dataset.")
    if cfg.start_step < 0 or (
        cfg.start_step + cfg.goal_offset_steps >= dataset.lengths[cfg.episode_index]
    ):
        raise ValueError("Selected task exceeds its episode.")

    chunk = dataset.load_chunk(
        np.asarray([cfg.episode_index]),
        np.asarray([cfg.start_step]),
        np.asarray([cfg.start_step + cfg.goal_offset_steps]),
    )[0]
    required = {"pixels", "state", "action"}
    missing = required - set(chunk)
    if missing:
        raise KeyError(f"Dataset task missing {sorted(missing)}")

    prep = get_img_preprocessor("pixels", "pixels", cfg.img_size)
    z_start = image_latent(runtime, chunk["pixels"][0], prep, device)
    z_goal = image_latent(runtime, chunk["pixels"][-1], prep, device)
    generator = torch.Generator(device=device).manual_seed(cfg.noise_seed)
    noise = torch.randn(
        1,
        cfg.candidates * cfg.rounds,
        cfg.horizon - 1,
        runtime.flow.latent_dim,
        device=device,
        generator=generator,
    )
    paths, costs = generate(
        runtime,
        encoder,
        scorer,
        z_start,
        z_goal,
        noise,
        cfg.candidates,
        cfg.rounds,
        cfg.experience_max_size,
        cfg.flow_steps,
    )
    # The source-episode bank provides an interpretable image projection for
    # raw planner and LeWM rollout latents. It is not used for any score.
    episode_chunk = dataset.load_chunk(
        np.asarray([cfg.episode_index]),
        np.asarray([0]),
        np.asarray([dataset.lengths[cfg.episode_index]]),
    )[0]
    bank_indices, bank_latents = episode_latent_bank(
        runtime,
        episode_chunk["pixels"],
        prep,
        device,
        cfg.retrieval_stride,
    )
    expert = expert_frames(chunk, cfg.horizon)
    all_actions = runtime.decode_actions(paths)
    lewm_paths = runtime.rollout_paths(
        z_start, all_actions, history_size=cfg.history_size
    )
    lewm_terminal_distances = (
        lewm_paths[:, :, -1] - z_goal[:, None]
    ).square().mean(dim=-1)[0]

    action_data = np.asarray(dataset.get_col_data("action"), dtype=np.float64).reshape(-1, 2)
    scaler = StandardScaler().fit(action_data[np.isfinite(action_data).all(axis=1)])
    seed, options = simulation_options(chunk)
    simulator_frames_all, simulator_terminal_distances = [], []
    for candidate_actions in all_actions[0]:
        frames, distance = simulator_rollout(
            candidate_actions, scaler, cfg.action_block, seed, options
        )
        simulator_frames_all.append(frames)
        simulator_terminal_distances.append(distance)
    simulator_terminal_distances = np.asarray(simulator_terminal_distances)

    output = cfg.output_dir / (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + f"_ep{cfg.episode_index}_step{cfg.start_step}"
    )
    output.mkdir(parents=True, exist_ok=False)
    expert_strip = frame_strip(expert, "Expert trajectory: dataset RGB frames", cfg.thumb_size)
    expert_strip.save(output / "expert_path_frames.png")

    count = min(cfg.num_rendered_paths, paths.size(1))
    rankings = {
        "raw_planner_cost": torch.argsort(costs[0])[:count].tolist(),
        "simulator_terminal_distance": np.argsort(simulator_terminal_distances)[:count].tolist(),
        "lewm_rollout_terminal_distance": torch.argsort(lewm_terminal_distances)[:count].tolist(),
    }
    rendered = {}
    for mode, selected_indices in rankings.items():
        overlays = [path_overlay(expert, "Expert path overlay")]
        strips = [expert_strip]
        records = []
        for rank, selected_index in enumerate(selected_indices, start=1):
            round_number = selected_index // cfg.candidates + 1
            raw_cost = float(costs[0, selected_index].item())
            simulator_distance = float(simulator_terminal_distances[selected_index])
            lewm_distance = float(lewm_terminal_distances[selected_index].item())

            if mode == "raw_planner_cost":
                visual_path = paths[0, selected_index]
                frames, nearest_indices, nearest_distances = nearest_episode_frames(
                    visual_path, bank_indices, bank_latents, episode_chunk["pixels"]
                )
                metric_label = f"LTC cost={raw_cost:.4f}"
                source_label = "Raw planner latent"
            elif mode == "simulator_terminal_distance":
                frames = simulator_frames_all[selected_index]
                nearest_indices, nearest_distances = None, None
                metric_label = f"true terminal distance={simulator_distance:.4f}"
                source_label = "Inverse dynamics + real PushT"
            else:
                visual_path = lewm_paths[0, selected_index]
                frames, nearest_indices, nearest_distances = nearest_episode_frames(
                    visual_path, bank_indices, bank_latents, episode_chunk["pixels"]
                )
                metric_label = f"LeWM terminal MSE={lewm_distance:.4f}"
                source_label = "Inverse dynamics + LeWM rollout"

            title = (
                f"{source_label} rank {rank}: candidate {selected_index}, "
                f"round {round_number}, {metric_label}"
            )
            strip = frame_strip(frames, title, cfg.thumb_size)
            strip.save(output / f"{mode}_rank_{rank:02d}_frames.png")
            strips.append(strip)
            overlays.append(path_overlay(frames, title))
            records.append(
                {
                    "rank": rank,
                    "candidate_index": selected_index,
                    "round": round_number,
                    "planner_path_cost": raw_cost,
                    "simulator_terminal_distance": simulator_distance,
                    "lewm_terminal_distance": lewm_distance,
                    "nearest_episode_frame_indices": (
                        None if nearest_indices is None else nearest_indices.tolist()
                    ),
                    "nearest_latent_distances": (
                        None if nearest_distances is None else nearest_distances.tolist()
                    ),
                }
            )

        stack_vertical(overlays).save(output / f"{mode}_overlay_comparison.png")
        stack_vertical(strips).save(output / f"{mode}_frame_strips_comparison.png")
        rendered[mode] = records

    np.savez_compressed(
        output / "candidate_metrics.npz",
        planner_paths=paths[0].detach().cpu().numpy(),
        planner_path_costs=costs[0].detach().cpu().numpy(),
        lewm_rollout_paths=lewm_paths[0].detach().cpu().numpy(),
        lewm_terminal_distances=lewm_terminal_distances.detach().cpu().numpy(),
        simulator_terminal_distances=simulator_terminal_distances,
        retrieval_episode_indices=bank_indices,
    )
    metadata = vars(cfg) | {
        "planner_checkpoint": str(cfg.planner_checkpoint),
        "output_dir": str(output),
        "ranking_results": rendered,
        "candidate_count": int(paths.size(1)),
        "retrieval_bank_size": int(len(bank_indices)),
        "note": (
            "Raw planner and LeWM panels are nearest-real-frame projections, "
            "whereas simulator panels are actual PushT action rollouts."
        ),
    }
    with (output / "metadata.json").open("w") as file:
        json.dump(metadata, file, indent=2)
    print(f"trajectory_rendering_complete output_dir={output}", flush=True)


if __name__ == "__main__":
    main()
