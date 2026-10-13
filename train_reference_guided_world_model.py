import os
os.environ.setdefault("MUJOCO_GL", "egl")

from pathlib import Path
import time
import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from torchvision.transforms import v2 as transforms

from latent_planner import LatentPlannerRuntime, stablewm_cache_dir
from reference_guided_world_model import (
    ReferenceGuidedWorldModel,
    reference_guided_training_step,
)
from utils import get_column_normalizer, get_img_preprocessor


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, index):
        item = self.dataset[index]
        episode, start = self.dataset.clip_indices[index]
        item["_episode_index"] = torch.tensor(episode)
        item["_start_step"] = torch.tensor(start)
        return item


def image_transform(img_size):
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=img_size),
    ])


def build_datasets(cfg):
    raw = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transformed = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    tx = [get_img_preprocessor("pixels", "pixels", cfg.img_size)]
    for col in cfg.data.dataset.keys_to_load:
        if col != "pixels":
            tx.append(get_column_normalizer(raw, col, col))
    transformed.transform = spt.data.transforms.Compose(*tx)
    # Match original planner training: loader clips are H+1 expert states,
    # separated by the planner action block. The first and last states form
    # the planner task; intermediate expert states are not used as labels here.
    return raw, IndexedDataset(transformed)


def make_loader(dataset, cfg, shuffle):
    return torch.utils.data.DataLoader(
        dataset, batch_size=cfg.loader.batch_size, shuffle=shuffle,
        drop_last=True, num_workers=cfg.loader.num_workers,
        pin_memory=True, persistent_workers=cfg.loader.num_workers > 0,
    )




def init_wandb(cfg, run_dir):
    if not cfg.wandb.enabled:
        return None
    try:
        import wandb
    except ImportError:
        print("WandB unavailable; continuing without it.", flush=True)
        return None
    return wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity or None,
        name=cfg.output_model_name,
        mode=cfg.wandb.mode,
        dir=str(run_dir),
        config=OmegaConf.to_container(cfg, resolve=True),
    )

def scalar(value):
    return float(value.detach()) if torch.is_tensor(value) else float(value)


def init_world_from_dataset(world, raw_dataset, episodes, starts, goal_offset):
    episodes = np.asarray(episodes, dtype=np.int64)
    starts = np.asarray(starts, dtype=np.int64)
    chunks = raw_dataset.load_chunk(episodes, starts, starts + int(goal_offset))
    seeds = []
    for chunk in chunks:
        seed = chunk.get("seed")
        seeds.append(None if seed is None else int(np.asarray(seed[0])))
    world.reset(seed=seeds, options=[{} for _ in range(len(chunks))])
    for env, chunk in zip(world.envs.unwrapped.envs, chunks):
        base = env.unwrapped
        if "state" not in chunk:
            raise KeyError("Real-rollout training requires a dataset state column.")
        base._set_state(np.asarray(chunk["state"][0]))
        # PushT restores its goal from the expert seed. Other supported
        # datasets may expose a goal_state column, in which case use it.
        if "goal_state" in chunk:
            base._set_goal_state(np.asarray(chunk["goal_state"][-1]))


@torch.no_grad()
def encode_world_pixels(runtime, world, transform, device):
    pixels = torch.as_tensor(world.infos["pixels"][:, -1])
    if pixels.ndim != 4:
        raise ValueError("Expected world pixels [B,H,W,C].")
    pixels = pixels.permute(0, 3, 1, 2)
    return runtime.encode_pixels(transform(pixels).to(device))


@torch.no_grad()
def simulator_targets(
    runtime, world, raw_dataset, episodes, starts, reference_paths, actions,
    *, goal_offset, image_tx, action_mean, action_std, device,
):
    batch, candidates, horizon, action_dim = actions.shape
    n = batch * candidates
    init_world_from_dataset(
        world, raw_dataset, np.repeat(episodes, candidates),
        np.repeat(starts, candidates), goal_offset,
    )
    base_dim = action_dim // runtime.action_block
    if base_dim * runtime.action_block != action_dim:
        raise ValueError("IDM action dimension must be divisible by action_block.")
    env_actions = actions.reshape(n, horizon, runtime.action_block, base_dim)
    env_actions = env_actions * action_std + action_mean
    targets = []
    for segment in range(horizon):
        for inner in range(runtime.action_block):
            result = world.envs.step(env_actions[:, segment, inner].detach().cpu().numpy())
            world.states, world.rewards, world.terminateds, world.truncateds, world.infos = result
            world.envs.unwrapped._autoreset_envs = np.zeros((n,), dtype=bool)
        targets.append(encode_world_pixels(runtime, world, image_tx, device))
    return torch.stack(targets, dim=1)


def checkpoint(model, runtime, cfg, path):
    torch.save({
        "reference_guided_world_model_state_dict": model.state_dict(),
        "architecture": model.architecture.__dict__,
        "planner_checkpoint": str(cfg.planner_checkpoint),
        "planner_action_block": runtime.action_block,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }, path)


@hydra.main(version_base=None, config_path="./config/train", config_name="reference_guided_world_model")
def run(cfg: DictConfig):
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    model, runtime = ReferenceGuidedWorldModel.from_planner_checkpoint(
        cfg.planner_checkpoint, device=device
    )
    runtime.eval().requires_grad_(False)
    payload = torch.load(cfg.planner_checkpoint, map_location="cpu", weights_only=False)
    planner_cfg = payload.get("config", {}).get("planner", {})
    if "horizon" not in planner_cfg:
        raise KeyError("Planner checkpoint has no config.planner.horizon metadata.")
    horizon = int(planner_cfg["horizon"])
    if horizon > model.architecture.max_horizon:
        raise ValueError("Planner training horizon exceeds checkpoint max_horizon.")
    # Mirror original planner clip construction from checkpoint metadata.
    with open_dict(cfg):
        cfg.data.dataset.frameskip = int(runtime.action_block)
        cfg.data.dataset.num_steps = horizon + 1

    raw_dataset, dataset = build_datasets(cfg)
    train_size = int(len(dataset) * cfg.train_split)
    train_set, val_set = torch.utils.data.random_split(
        dataset, [train_size, len(dataset) - train_size],
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    train_loader, val_loader = make_loader(train_set, cfg, True), make_loader(val_set, cfg, False)

    action_data = torch.as_tensor(raw_dataset.get_col_data("action"), dtype=torch.float32)
    action_data = action_data[~torch.isnan(action_data).any(dim=1)]
    action_mean, action_std = action_data.mean(0).to(device), action_data.std(0).clamp_min(1e-6).to(device)
    world_cfg = OmegaConf.to_container(cfg.world, resolve=True)
    world_cfg["num_envs"] = cfg.loader.batch_size
    world_cfg["max_episode_steps"] = max(world_cfg["max_episode_steps"], horizon * runtime.action_block)
    world = swm.World(**world_cfg, image_shape=(224, 224))
    pixel_tx = image_transform(cfg.img_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)

    run_dir = Path(stablewm_cache_dir(sub_folder="checkpoints"), cfg.subdir)
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "reference_guided_world_model_config.yaml")
    wandb_run = init_wandb(cfg, run_dir)
    global_step = 0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for batch_index, batch in enumerate(train_loader):
            with torch.no_grad():
                z_expert = runtime.encode_pixels(batch["pixels"].to(device))
                z_start, z_goal = z_expert[:, 0], z_expert[:, -1]
                paths = runtime.sample_paths(z_start, z_goal, horizon=horizon,
                    num_samples=1, flow_steps=cfg.sampling.flow_steps)
                actions = runtime.decode_actions(paths)
                target = simulator_targets(
                    runtime, world, raw_dataset, batch["_episode_index"].numpy(),
                    batch["_start_step"].numpy(), paths, actions,
                    goal_offset=horizon * runtime.action_block, image_tx=pixel_tx,
                    action_mean=action_mean, action_std=action_std, device=device,
                )
            flat_paths = paths.reshape(-1, horizon + 1, paths.size(-1))
            flat_actions = actions.reshape(-1, horizon, actions.size(-1))
            out = reference_guided_training_step(model, runtime.lewm, flat_paths, flat_actions, target)
            optimizer.zero_grad(set_to_none=True)
            out["loss"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()
            global_step += 1
            if global_step % cfg.log_interval == 0:
                print(f"epoch={epoch} step={global_step} loss={scalar(out['loss']):.6f} grad_norm={scalar(grad_norm):.4f}", flush=True)
            if wandb_run is not None:
                wandb_run.log({"train/loss": scalar(out["loss"]), "train/grad_norm": scalar(grad_norm), "train/epoch": epoch}, step=global_step)
            if cfg.max_train_batches and batch_index + 1 >= cfg.max_train_batches:
                break
        checkpoint(model, runtime, cfg, run_dir / f"{cfg.output_model_name}_epoch_{epoch}.pt")
        checkpoint(model, runtime, cfg, run_dir / f"{cfg.output_model_name}.pt")
        print(f"epoch={epoch} checkpoint_done", flush=True)
    if wandb_run is not None:
        wandb_run.finish()
    world.close()


if __name__ == "__main__":
    run()
