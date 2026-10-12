import json
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from latent_planner import (
    LatentPathFlow,
    checkpoint_payload,
    joint_flow_matching_loss,
    lewm_consistency_loss,
    lewm_segment_feasibility_losses,
    load_world_model,
    smoothness_loss,
    stablewm_cache_dir,
)
from utils import get_column_normalizer, get_img_preprocessor


def init_wandb(cfg: DictConfig, run_dir: Path):
    if not cfg.wandb.enabled:
        return None
    try:
        import wandb
    except ImportError:
        print("WandB requested but not installed; continuing without WandB.")
        return None

    wandb_cfg = OmegaConf.to_container(cfg, resolve=True)
    return wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity or None,
        name=cfg.wandb.name or cfg.subdir.replace("/", "_"),
        id=cfg.wandb.id or None,
        resume=cfg.wandb.resume,
        mode=cfg.wandb.mode,
        dir=str(run_dir),
        config=wandb_cfg,
    )


def metric_float(value) -> float:
    if torch.is_tensor(value):
        value = value.detach()
    return float(value)


def freeze(module: torch.nn.Module) -> torch.nn.Module:
    module.eval()
    module.requires_grad_(False)
    return module


def default_episode_split_path(cfg: DictConfig) -> Path:
    dataset_name = str(cfg.data.dataset.name).replace("/", "_")
    pct = int(round(float(cfg.episode_split.train_fraction) * 100))
    filename = f"{dataset_name}_seed{cfg.episode_split.seed}_train{pct}.json"
    return Path(stablewm_cache_dir(), "splits", filename)


def load_or_create_episode_split(cfg: DictConfig, num_episodes: int) -> tuple[set[int], set[int], Path]:
    split_file = (
        Path(cfg.episode_split.split_file)
        if cfg.episode_split.split_file
        else default_episode_split_path(cfg)
    )
    split_file.parent.mkdir(parents=True, exist_ok=True)

    if split_file.exists():
        with split_file.open("r") as f:
            payload = json.load(f)
    else:
        rng = np.random.default_rng(int(cfg.episode_split.seed))
        episodes = np.arange(num_episodes)
        rng.shuffle(episodes)
        n_train = int(round(num_episodes * float(cfg.episode_split.train_fraction)))
        n_train = min(max(n_train, 1), max(num_episodes - 1, 1))
        payload = {
            "dataset": str(cfg.data.dataset.name),
            "seed": int(cfg.episode_split.seed),
            "train_fraction": float(cfg.episode_split.train_fraction),
            "num_episodes": int(num_episodes),
            "train_episodes": sorted(int(x) for x in episodes[:n_train]),
            "eval_episodes": sorted(int(x) for x in episodes[n_train:]),
        }
        with split_file.open("w") as f:
            json.dump(payload, f, indent=2)

    train_episodes = {int(x) for x in payload["train_episodes"]}
    eval_episodes = {int(x) for x in payload["eval_episodes"]}
    if not train_episodes or not eval_episodes:
        raise ValueError(f"Episode split at {split_file} must contain train and eval episodes.")
    return train_episodes, eval_episodes, split_file


@torch.no_grad()
def encode_latents(lewm: torch.nn.Module, batch: dict, device: torch.device) -> torch.Tensor:
    return lewm.encode({"pixels": batch["pixels"].to(device)})["emb"].detach()


def make_loaders(cfg: DictConfig):
    dataset = swm.data.HDF5Dataset(**cfg.data.dataset, transform=None)
    transforms = [
        get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)
    ]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith("pixels"):
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)
            setattr(cfg.data, f"{col}_dim", dataset.get_dim(col))

    dataset.transform = spt.data.transforms.Compose(*transforms)

    if cfg.episode_split.enabled:
        train_episodes, eval_episodes, split_file = load_or_create_episode_split(
            cfg, num_episodes=len(dataset.lengths)
        )
        dataset.clip_indices = [
            (ep, start) for ep, start in dataset.clip_indices if int(ep) in train_episodes
        ]
        print(
            "episode_split enabled "
            f"split_file={split_file} train_episodes={len(train_episodes)} "
            f"heldout_episodes={len(eval_episodes)} train_clips={len(dataset)}",
            flush=True,
        )
        if len(dataset) == 0:
            raise ValueError(
                f"Episode split {split_file} produced zero train clips for {cfg.data.dataset.name}."
            )

    gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset, lengths=[cfg.train_split, 1 - cfg.train_split], generator=gen
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        **cfg.loader,
        shuffle=True,
        drop_last=True,
        generator=gen,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        **cfg.loader,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, val_loader


def step_batch(
    *,
    batch: dict,
    lewm: torch.nn.Module,
    flow: LatentPathFlow,
    cfg: DictConfig,
    device: torch.device,
    feasibility_scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    batch["action"] = torch.nan_to_num(batch["action"].to(device), 0.0)
    z_path = encode_latents(lewm, batch, device)
    actions = batch["action"][:, : cfg.planner.horizon]
    loss_flow, loss_action, pred_path, pred_actions = joint_flow_matching_loss(
        flow, z_path, actions
    )

    if (
        cfg.loss.feasibility.action_weight
        or cfg.loss.feasibility.path_weight
    ):
        loss_feasible_action, loss_feasible_path = lewm_segment_feasibility_losses(
            lewm,
            pred_path,
            pred_actions,
        )
    else:
        loss_feasible_action = z_path.new_tensor(0.0)
        loss_feasible_path = z_path.new_tensor(0.0)

    if cfg.loss.consistency.weight:
        if cfg.loss.consistency.detach_actions:
            with torch.no_grad():
                loss_dyn = lewm_consistency_loss(
                    lewm,
                    z_path,
                    pred_actions.detach(),
                    history_size=cfg.lewm_history_size,
                )
        else:
            loss_dyn = lewm_consistency_loss(
                lewm,
                z_path,
                pred_actions,
                history_size=cfg.lewm_history_size,
            )
    else:
        loss_dyn = z_path.new_tensor(0.0)
    loss_smooth = smoothness_loss(z_path)
    total = (
        cfg.loss.flow.weight * loss_flow
        + cfg.loss.action.weight * loss_action
        + cfg.loss.consistency.weight * loss_dyn
        + cfg.loss.smoothness.weight * loss_smooth
        + feasibility_scale
        * (
            cfg.loss.feasibility.action_weight * loss_feasible_action
            + cfg.loss.feasibility.path_weight * loss_feasible_path
        )
    )
    return {
        "loss": total,
        "flow_loss": loss_flow.detach(),
        "action_loss": loss_action.detach(),
        "consistency_loss": loss_dyn.detach(),
        "feasibility_action_loss": loss_feasible_action.detach(),
        "feasibility_path_loss": loss_feasible_path.detach(),
        "feasibility_scale": z_path.new_tensor(feasibility_scale),
        "smoothness_loss": loss_smooth.detach(),
    }


@torch.no_grad()
def validate(
    *,
    loader,
    lewm: torch.nn.Module,
    flow: LatentPathFlow,
    cfg: DictConfig,
    device: torch.device,
) -> dict[str, float]:
    flow.eval()
    sums: dict[str, float] = {}
    count = 0
    for i, batch in enumerate(loader):
        out = step_batch(
            batch=batch,
            lewm=lewm,
            flow=flow,
            cfg=cfg,
            device=device,
        )
        bs = batch["pixels"].size(0)
        count += bs
        for k, v in out.items():
            sums[k] = sums.get(k, 0.0) + float(v) * bs
        if cfg.val_batches is not None and i + 1 >= cfg.val_batches:
            break
    flow.train()
    return {k: v / max(count, 1) for k, v in sums.items()}


@hydra.main(version_base=None, config_path="./config/train", config_name="latent_planner")
def run(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = make_loaders(cfg)
    lewm = freeze(
        load_world_model(
            cfg.lewm_checkpoint,
            world_model_type=cfg.world_model.type,
            fast_source_dir=cfg.world_model.fast_source_dir,
        ).to(device)
    )

    first_batch = next(iter(train_loader))
    z = encode_latents(lewm, first_batch, device)
    latent_dim = z.size(-1)
    action_dim = first_batch["action"].size(-1)

    flow = LatentPathFlow(
        latent_dim=latent_dim,
        action_dim=action_dim,
        max_horizon=cfg.planner.max_horizon,
        **cfg.flow,
    ).to(device)

    optimizer = torch.optim.AdamW(flow.parameters(), **cfg.optimizer)

    run_dir = Path(stablewm_cache_dir(sub_folder="checkpoints"), cfg.subdir)
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "latent_planner_config.yaml")
    wandb_run = init_wandb(cfg, run_dir)

    global_step = 0
    try:
        for epoch in range(cfg.epochs):
            print(f"epoch={epoch + 1} train_start", flush=True)
            flow.train()
            for batch_idx, batch in enumerate(train_loader):
                out = step_batch(
                    batch=batch,
                    lewm=lewm,
                    flow=flow,
                    cfg=cfg,
                    device=device,
                    feasibility_scale=min(
                        1.0,
                        (global_step + 1)
                        / max(int(cfg.loss.feasibility.warmup_steps), 1),
                    ),
                )
                optimizer.zero_grad(set_to_none=True)
                out["loss"].backward()
                grad_norm = None
                if cfg.grad_clip_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        flow.parameters(),
                        cfg.grad_clip_norm,
                    )
                optimizer.step()
                global_step += 1

                if global_step % cfg.log_interval == 0:
                    metrics = " ".join(
                        f"{k}={metric_float(v):.4f}" for k, v in out.items()
                    )
                    print(f"epoch={epoch + 1} step={global_step} {metrics}", flush=True)

                if wandb_run is not None:
                    log_data = {
                        f"train/{k}": metric_float(v)
                        for k, v in out.items()
                    }
                    log_data["train/epoch"] = epoch + 1
                    log_data["train/lr"] = optimizer.param_groups[0]["lr"]
                    if grad_norm is not None:
                        log_data["train/grad_norm"] = float(grad_norm)
                    wandb_run.log(log_data, step=global_step)

                if cfg.max_train_batches is not None and batch_idx + 1 >= cfg.max_train_batches:
                    break

            print(f"epoch={epoch + 1} validation_start", flush=True)
            val_metrics = validate(
                loader=val_loader,
                lewm=lewm,
                flow=flow,
                cfg=cfg,
                device=device,
            )
            print(
                f"epoch={epoch + 1} validation "
                + " ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()),
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log(
                    {f"val/{k}": v for k, v in val_metrics.items()}
                    | {"val/epoch": epoch + 1},
                    step=global_step,
                )

            payload = checkpoint_payload(
                lewm_checkpoint=str(cfg.lewm_checkpoint),
                world_model_type=str(cfg.world_model.type),
                fast_source_dir=(
                    str(cfg.world_model.fast_source_dir)
                    if cfg.world_model.fast_source_dir is not None
                    else None
                ),
                action_block=cfg.planner.action_block,
                flow=flow,
                inverse_dynamics=None,
                cfg=OmegaConf.to_container(cfg, resolve=True),
            )
            epoch_path = run_dir / f"{cfg.output_model_name}_epoch_{epoch + 1}.pt"
            latest_path = run_dir / f"{cfg.output_model_name}.pt"
            print(f"epoch={epoch + 1} checkpoint_start path={epoch_path}", flush=True)
            torch.save(payload, epoch_path)
            torch.save(payload, latest_path)
            print(f"epoch={epoch + 1} checkpoint_done path={latest_path}", flush=True)
            if wandb_run is not None and cfg.wandb.log_model:
                wandb_run.save(str(latest_path), base_path=str(run_dir))
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    run()
