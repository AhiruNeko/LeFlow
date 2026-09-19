import json
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from latent_planner import (
    InverseDynamics,
    LatentPathFlow,
    checkpoint_payload,
    flow_matching_loss,
    inverse_dynamics_loss,
    lewm_consistency_loss,
    load_lewm,
    smoothness_loss,
    stablewm_cache_dir,
)
from latent_trajectory_cost import TrajectoryCostModel, TrajectoryEncoder
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


def _checkpoint_load(path: str | Path) -> dict[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.0
        return torch.load(path, map_location="cpu")


def load_ltc_components(
    cfg: DictConfig, latent_dim: int, device: torch.device
) -> tuple[TrajectoryEncoder, TrajectoryCostModel]:
    """Load the separately saved LTC encoder and cost model."""
    encoder_payload = _checkpoint_load(cfg.experience.trajectory_encoder_checkpoint)
    cost_payload = _checkpoint_load(cfg.experience.cost_model_checkpoint)
    if encoder_payload.get("component") != "trajectory_encoder":
        raise ValueError("experience.trajectory_encoder_checkpoint is not an LTC encoder checkpoint")
    if cost_payload.get("component") != "trajectory_cost":
        raise ValueError("experience.cost_model_checkpoint is not an LTC cost checkpoint")

    trajectory_encoder = TrajectoryEncoder(
        latent_dim=latent_dim, **dict(encoder_payload["architecture"])
    ).to(device)
    trajectory_encoder.load_state_dict(encoder_payload["state_dict"], strict=True)
    cost_model = TrajectoryCostModel(**dict(cost_payload["architecture"])).to(device)
    cost_model.load_state_dict(cost_payload["state_dict"], strict=True)
    return trajectory_encoder, cost_model


def save_finetuned_ltc_components(
    *,
    run_dir: Path,
    output_model_name: str,
    epoch: int,
    lewm_checkpoint: str,
    trajectory_encoder: TrajectoryEncoder,
    cost_model: TrajectoryCostModel,
    cfg: DictConfig,
) -> tuple[Path, Path]:
    """Save fine-tuned LTC components in the same independently loadable format."""
    encoder_arch = {
        "max_horizon": trajectory_encoder.max_horizon,
        "model_dim": trajectory_encoder.cls_token.size(-1),
        "representation_dim": trajectory_encoder.representation_dim,
        "depth": len(trajectory_encoder.blocks),
        "heads": trajectory_encoder.blocks[0].attn.num_heads,
        "mlp_dim": trajectory_encoder.blocks[0].mlp[0].out_features,
        "dropout": trajectory_encoder.blocks[0].attn.dropout,
    }
    cost_arch = {
        "representation_dim": cost_model.representation_dim,
        "dropout": cost_model.network[3].p,
    }
    common = {
        "format": "latent_trajectory_cost_component_v1",
        "lewm_checkpoint": lewm_checkpoint,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    encoder_payload = common | {
        "component": "trajectory_encoder",
        "architecture": encoder_arch,
        "state_dict": trajectory_encoder.state_dict(),
    }
    cost_payload = common | {
        "component": "trajectory_cost",
        "architecture": cost_arch,
        "state_dict": cost_model.state_dict(),
    }
    encoder_path = run_dir / f"{output_model_name}_trajectory_encoder.pt"
    cost_path = run_dir / f"{output_model_name}_cost_model.pt"
    torch.save(encoder_payload, run_dir / f"{output_model_name}_trajectory_encoder_epoch_{epoch}.pt")
    torch.save(cost_payload, run_dir / f"{output_model_name}_cost_model_epoch_{epoch}.pt")
    torch.save(encoder_payload, encoder_path)
    torch.save(cost_payload, cost_path)
    return encoder_path, cost_path


def noised_path_all_after_start(path: torch.Tensor, noise_std: float) -> torch.Tensor:
    """Perturb every state after z_0, including the candidate endpoint."""
    latent_scale = path.detach().std(unbiased=False).clamp_min(1e-6)
    result = path.clone()
    result[:, 1:] += torch.randn_like(result[:, 1:]) * (noise_std * latent_scale)
    return result


def noised_path_intermediate(path: torch.Tensor, noise_std: float) -> torch.Tensor:
    """Perturb intermediate states only; preserve z_0 and the endpoint."""
    if path.size(1) < 3:
        raise ValueError("intermediate-noise experiences require at least three path states")
    latent_scale = path.detach().std(unbiased=False).clamp_min(1e-6)
    result = path.clone()
    result[:, 1:-1] += torch.randn_like(result[:, 1:-1]) * (noise_std * latent_scale)
    return result


@torch.no_grad()
def build_experience_bank(
    *,
    z_path: torch.Tensor,
    trajectory_encoder: TrajectoryEncoder,
    cost_model: TrajectoryCostModel,
    cfg: DictConfig,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Build a same-task, synthetic-noise experience bank for phase one.

    Every memory trajectory is paired with the query's own start and true goal.
    Half are expected to have all states after z_0 perturbed (including their
    endpoint); the other half preserve their endpoint and perturb only the
    intermediate states.  No expert positive path and no goal mismatch is put
    into the memory, so the planner cannot retrieve its supervision target.
    """
    batch_size, _, latent_dim = z_path.shape
    min_size = int(cfg.experience.min_size)
    max_size = int(cfg.experience.max_size)
    if min_size < 0 or max_size < min_size:
        raise ValueError("experience sizes must satisfy 0 <= min_size <= max_size")
    size = int(torch.randint(min_size, max_size + 1, (), device=z_path.device))
    if size == 0:
        return None, None

    repeated_path = z_path[:, None].expand(-1, size, -1, -1).reshape(
        batch_size * size, z_path.size(1), latent_dim
    )
    all_after_start = noised_path_all_after_start(
        repeated_path, float(cfg.experience.noise_std)
    )
    intermediate_only = noised_path_intermediate(
        repeated_path, float(cfg.experience.noise_std)
    )
    choose_all_after_start = torch.rand(
        batch_size * size, device=z_path.device
    ) < float(cfg.experience.all_after_start_probability)
    candidate_paths = torch.where(
        choose_all_after_start[:, None, None], all_after_start, intermediate_only
    )
    candidate_goals = z_path[:, -1, :].repeat_interleave(size, dim=0)

    features = trajectory_encoder(candidate_paths, candidate_goals)
    costs = cost_model(features)
    return (
        features.reshape(batch_size, size, -1),
        costs.reshape(batch_size, size),
    )


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
    inverse_dynamics: InverseDynamics,
    trajectory_encoder: TrajectoryEncoder,
    cost_model: TrajectoryCostModel,
    cfg: DictConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Train on an expert path while conditioning the flow on LTC experiences."""
    batch["action"] = torch.nan_to_num(batch["action"].to(device), 0.0)
    z_path = encode_latents(lewm, batch, device)
    actions = batch["action"][:, : cfg.planner.horizon]
    path_features, path_costs = build_experience_bank(
        z_path=z_path,
        trajectory_encoder=trajectory_encoder,
        cost_model=cost_model,
        cfg=cfg,
    )

    loss_flow = flow_matching_loss(
        flow, z_path, path_features=path_features, path_costs=path_costs
    )
    loss_inv, pred_actions = inverse_dynamics_loss(inverse_dynamics, z_path, actions)
    if cfg.loss.consistency.weight:
        if cfg.loss.consistency.detach_inverse:
            with torch.no_grad():
                loss_dyn = lewm_consistency_loss(
                    lewm, z_path, pred_actions.detach(), history_size=cfg.lewm_history_size
                )
        else:
            loss_dyn = lewm_consistency_loss(
                lewm, z_path, pred_actions, history_size=cfg.lewm_history_size
            )
    else:
        loss_dyn = z_path.new_tensor(0.0)
    loss_smooth = smoothness_loss(z_path)
    total = (
        cfg.loss.flow.weight * loss_flow
        + cfg.loss.inverse.weight * loss_inv
        + cfg.loss.consistency.weight * loss_dyn
        + cfg.loss.smoothness.weight * loss_smooth
    )
    return {
        "loss": total,
        "flow_loss": loss_flow.detach(),
        "inverse_loss": loss_inv.detach(),
        "consistency_loss": loss_dyn.detach(),
        "smoothness_loss": loss_smooth.detach(),
        "experience_size": z_path.new_tensor(
            0 if path_features is None else path_features.size(1)
        ),
        "experience_cost": (
            z_path.new_zeros(())
            if path_costs is None
            else path_costs.detach().mean()
        ),
    }

@torch.no_grad()
def validate(
    *,
    loader,
    lewm: torch.nn.Module,
    flow: LatentPathFlow,
    inverse_dynamics: InverseDynamics,
    trajectory_encoder: TrajectoryEncoder,
    cost_model: TrajectoryCostModel,
    cfg: DictConfig,
    device: torch.device,
) -> dict[str, float]:
    flow.eval()
    inverse_dynamics.eval()
    trajectory_encoder.eval()
    cost_model.eval()
    sums: dict[str, float] = {}
    count = 0
    for i, batch in enumerate(loader):
        out = step_batch(
            batch=batch,
            lewm=lewm,
            flow=flow,
            inverse_dynamics=inverse_dynamics,
            trajectory_encoder=trajectory_encoder,
            cost_model=cost_model,
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
    inverse_dynamics.train()
    trajectory_encoder.eval()
    cost_model.eval()
    return {k: v / max(count, 1) for k, v in sums.items()}


@hydra.main(version_base=None, config_path="./config/train", config_name="latent_planner")
def run(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = make_loaders(cfg)
    lewm = freeze(load_lewm(cfg.lewm_checkpoint).to(device))

    first_batch = next(iter(train_loader))
    z = encode_latents(lewm, first_batch, device)
    latent_dim = z.size(-1)
    action_dim = first_batch["action"].size(-1)

    flow = LatentPathFlow(
        latent_dim=latent_dim,
        max_horizon=cfg.planner.max_horizon,
        **cfg.flow,
    ).to(device)
    inverse_dynamics = InverseDynamics(
        latent_dim=latent_dim,
        action_dim=action_dim,
        **cfg.inverse_dynamics,
    ).to(device)
    trajectory_encoder, cost_model = load_ltc_components(cfg, latent_dim, device)
    if trajectory_encoder.representation_dim != flow.path_feature_dim:
        raise ValueError(
            "LTC representation_dim must equal flow.path_feature_dim; got "
            f"{trajectory_encoder.representation_dim} and {flow.path_feature_dim}."
        )

    # Phase one keeps LTC fixed: it supplies a stable cost-conditioned memory
    # representation while only the planner and inverse dynamics learn.
    trajectory_encoder = freeze(trajectory_encoder)
    cost_model = freeze(cost_model)
    trainable_parameters = list(flow.parameters()) + list(inverse_dynamics.parameters())
    optimizer = torch.optim.AdamW(trainable_parameters, **cfg.optimizer)

    run_dir = Path(stablewm_cache_dir(sub_folder="checkpoints"), cfg.subdir)
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "latent_planner_config.yaml")
    wandb_run = init_wandb(cfg, run_dir)

    global_step = 0
    try:
        for epoch in range(cfg.epochs):
            print(f"epoch={epoch + 1} train_start", flush=True)
            flow.train()
            inverse_dynamics.train()
            trajectory_encoder.eval()
            cost_model.eval()
            for batch_idx, batch in enumerate(train_loader):
                out = step_batch(
                    batch=batch,
                    lewm=lewm,
                    flow=flow,
                    inverse_dynamics=inverse_dynamics,
                    trajectory_encoder=trajectory_encoder,
                    cost_model=cost_model,
                    cfg=cfg,
                    device=device,
                )
                optimizer.zero_grad(set_to_none=True)
                out["loss"].backward()
                grad_norm = None
                if cfg.grad_clip_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        trainable_parameters,
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
                inverse_dynamics=inverse_dynamics,
                trajectory_encoder=trajectory_encoder,
                cost_model=cost_model,
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
                action_block=cfg.planner.action_block,
                flow=flow,
                inverse_dynamics=inverse_dynamics,
                cfg=OmegaConf.to_container(cfg, resolve=True),
            )
            payload["experience"] = {
                "trajectory_encoder_checkpoint": str(cfg.experience.trajectory_encoder_checkpoint),
                "cost_model_checkpoint": str(cfg.experience.cost_model_checkpoint),
                "trajectory_encoder_state_dict": trajectory_encoder.state_dict(),
                "cost_model_state_dict": cost_model.state_dict(),
            }
            epoch_path = run_dir / f"{cfg.output_model_name}_epoch_{epoch + 1}.pt"
            latest_path = run_dir / f"{cfg.output_model_name}.pt"
            print(f"epoch={epoch + 1} checkpoint_start path={epoch_path}", flush=True)
            torch.save(payload, epoch_path)
            torch.save(payload, latest_path)
            encoder_path, cost_path = save_finetuned_ltc_components(
                run_dir=run_dir,
                output_model_name=cfg.output_model_name,
                epoch=epoch + 1,
                lewm_checkpoint=str(cfg.lewm_checkpoint),
                trajectory_encoder=trajectory_encoder,
                cost_model=cost_model,
                cfg=cfg,
            )
            print(
                f"epoch={epoch + 1} checkpoint_done path={latest_path} "
                f"encoder={encoder_path} cost={cost_path}",
                flush=True,
            )
            if wandb_run is not None and cfg.wandb.log_model:
                wandb_run.save(str(latest_path), base_path=str(run_dir))
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    run()
