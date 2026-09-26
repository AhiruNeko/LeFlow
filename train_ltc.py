"""Train a goal-conditioned trajectory encoder and latent trajectory cost (LTC).

The frozen LeWM encodes expert image paths into latent trajectories.  Each expert
trajectory is a positive example.  Two synthetic negative examples are built for
every positive path: (1) the same path with a goal from another batch element,
and (2) a path whose intermediate latent states are noised while its endpoints
are left unchanged.  The encoder and cost predictor are trained with the
pairwise logistic preference objective used by Traj-LeWM.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from latent_planner import load_lewm, stablewm_cache_dir
from module import SIGReg
from latent_trajectory_cost import TrajectoryCostModel, TrajectoryEncoder
from train_latent_planner import (
    encode_latents,
    freeze,
    init_wandb,
    make_loaders,
    metric_float,
)


def preference_loss(
    positive_cost: torch.Tensor,
    negative_cost: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """Encourage every positive path to have lower cost than its negative.

    Implements ``log(1 + exp(-(c_neg - c_pos) / beta))``.  Lower cost denotes a
    more desirable trajectory.
    """
    if beta <= 0:
        raise ValueError("preference beta must be positive")
    return F.softplus(-(negative_cost - positive_cost) / beta).mean()


def representation_sigreg(
    sigreg: SIGReg | None,
    *representations: torch.Tensor,
) -> torch.Tensor:
    """Regularize the joint expert/negative representation distribution."""
    if sigreg is None:
        return representations[0].new_zeros(())
    features = torch.cat(representations, dim=0)
    return sigreg(features.unsqueeze(0))


def mismatched_goals(goals: torch.Tensor) -> torch.Tensor:
    """Assign every expert path another sample's final goal.

    A cyclic shift guarantees a mismatch when the batch contains at least two
    samples and avoids a CPU round-trip during training.
    """
    if goals.size(0) < 2:
        raise ValueError("goal-mismatch training requires batch_size >= 2")
    shift = int(torch.randint(1, goals.size(0), (), device=goals.device))
    return goals.roll(shifts=shift, dims=0)


def noised_path(path: torch.Tensor, noise_std: float) -> torch.Tensor:
    """Perturb only intermediate states; preserve path start and endpoint."""
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")
    if path.size(1) < 3:
        raise ValueError("noised negative paths require at least three states")

    # Dataset/batch-adaptive scale keeps the noise magnitude meaningful if the
    # LeWM latent representation changes across datasets or checkpoints.
    latent_scale = path.detach().std(unbiased=False).clamp_min(1e-6)
    perturbation = torch.randn_like(path[:, 1:-1]) * (noise_std * latent_scale)
    negative = path.clone()
    negative[:, 1:-1] = negative[:, 1:-1] + perturbation
    return negative


def ltc_outputs(
    encoder: TrajectoryEncoder,
    cost_model: TrajectoryCostModel,
    path: torch.Tensor,
    goal: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    representation = encoder(path, goal)
    return representation, cost_model(representation)


def step_batch(
    *,
    batch: dict[str, torch.Tensor],
    lewm: torch.nn.Module,
    trajectory_encoder: TrajectoryEncoder,
    cost_model: TrajectoryCostModel,
    sigreg: SIGReg | None,
    cfg: DictConfig,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Build the positive and two synthetic negative path pairs for one batch."""
    positive_path = encode_latents(lewm, batch, device)
    positive_goal = positive_path[:, -1]

    positive_representation, positive_cost = ltc_outputs(
        trajectory_encoder, cost_model, positive_path, positive_goal
    )

    mismatch_goal = mismatched_goals(positive_goal)
    mismatch_representation, mismatch_cost = ltc_outputs(
        trajectory_encoder, cost_model, positive_path, mismatch_goal
    )
    goal_mismatch_loss = preference_loss(
        positive_cost, mismatch_cost, float(cfg.preference.beta)
    )

    noisy_negative_path = noised_path(
        positive_path, float(cfg.negatives.noise_std)
    )
    noisy_representation, noisy_cost = ltc_outputs(
        trajectory_encoder, cost_model, noisy_negative_path, positive_goal
    )
    noise_loss = preference_loss(positive_cost, noisy_cost, float(cfg.preference.beta))
    sigreg_loss = representation_sigreg(
        sigreg,
        positive_representation,
        mismatch_representation,
        noisy_representation,
    )

    total = (
        float(cfg.negatives.goal_mismatch_weight) * goal_mismatch_loss
        + float(cfg.negatives.noise_weight) * noise_loss
        + float(cfg.loss.sigreg.weight) * sigreg_loss
    )
    metrics = {
        "loss": total,
        "goal_mismatch_loss": goal_mismatch_loss.detach(),
        "noise_loss": noise_loss.detach(),
        "sigreg_loss": sigreg_loss.detach(),
        "positive_cost": positive_cost.detach().mean(),
        "mismatch_cost": mismatch_cost.detach().mean(),
        "noisy_cost": noisy_cost.detach().mean(),
        "mismatch_margin": (mismatch_cost - positive_cost).detach().mean(),
        "noise_margin": (noisy_cost - positive_cost).detach().mean(),
    }
    cost_distributions = {
        "expert": positive_cost.detach(),
        "goal_mismatch": mismatch_cost.detach(),
        "noisy": noisy_cost.detach(),
    }
    return metrics, cost_distributions


@torch.no_grad()
def validate(
    *,
    loader: Any,
    lewm: torch.nn.Module,
    trajectory_encoder: TrajectoryEncoder,
    cost_model: TrajectoryCostModel,
    sigreg: SIGReg | None,
    cfg: DictConfig,
    device: torch.device,
) -> dict[str, float]:
    trajectory_encoder.eval()
    cost_model.eval()
    sums: dict[str, float] = {}
    count = 0
    for batch_index, batch in enumerate(loader):
        output, _ = step_batch(
            batch=batch,
            lewm=lewm,
            trajectory_encoder=trajectory_encoder,
            cost_model=cost_model,
            sigreg=sigreg,
            cfg=cfg,
            device=device,
        )
        batch_size = batch["pixels"].size(0)
        count += batch_size
        for name, value in output.items():
            sums[name] = sums.get(name, 0.0) + metric_float(value) * batch_size
        if cfg.val_batches is not None and batch_index + 1 >= cfg.val_batches:
            break
    trajectory_encoder.train()
    cost_model.train()
    return {name: value / max(count, 1) for name, value in sums.items()}


def ltc_checkpoint_payload(
    *,
    trajectory_encoder: TrajectoryEncoder,
    cost_model: TrajectoryCostModel,
    cfg: DictConfig,
) -> dict[str, Any]:
    """Serialize the encoder and cost model as one inseparable LTC checkpoint."""
    return {
        "format": "latent_trajectory_cost_v2",
        "lewm_checkpoint": str(cfg.lewm_checkpoint),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "trajectory_encoder": {
            "architecture": OmegaConf.to_container(
                cfg.trajectory_encoder,
                resolve=True,
            ),
            "state_dict": trajectory_encoder.state_dict(),
        },
        "cost_model": {
            "architecture": {
                "representation_dim": int(
                    cfg.trajectory_encoder.representation_dim
                ),
                "dropout": float(cfg.trajectory_encoder.dropout),
            },
            "state_dict": cost_model.state_dict(),
        },
    }


def save_ltc_checkpoint(
    *,
    run_dir: Path,
    epoch: int,
    trajectory_encoder: TrajectoryEncoder,
    cost_model: TrajectoryCostModel,
    cfg: DictConfig,
) -> Path:
    """Save both trained LTC modules in a single checkpoint file."""
    payload = ltc_checkpoint_payload(
        trajectory_encoder=trajectory_encoder,
        cost_model=cost_model,
        cfg=cfg,
    )
    epoch_path = run_dir / f"{cfg.output_model_name}_epoch_{epoch}.pt"
    latest_path = run_dir / f"{cfg.output_model_name}.pt"
    torch.save(payload, epoch_path)
    torch.save(payload, latest_path)
    return latest_path


def cost_distribution_wandb_data(
    costs: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Build W&B histograms and quantiles for expert and negative costs."""
    import wandb

    log_data: dict[str, Any] = {}
    for name, values in costs.items():
        values = values.detach().float().flatten().cpu()
        log_data[f"train/cost_distribution/{name}_histogram"] = wandb.Histogram(
            values.numpy()
        )
        for quantile in (0.1, 0.5, 0.9):
            label = f"q{int(quantile * 100):02d}"
            log_data[f"train/cost_distribution/{name}_{label}"] = float(
                torch.quantile(values, quantile)
            )
    return log_data


@hydra.main(version_base=None, config_path="./config/train", config_name="ltc")
def run(cfg: DictConfig) -> None:
    torch.manual_seed(int(cfg.seed))
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = make_loaders(cfg)
    lewm = freeze(load_lewm(cfg.lewm_checkpoint).to(device))

    first_batch = next(iter(train_loader))
    first_path = encode_latents(lewm, first_batch, device)
    latent_dim = first_path.size(-1)
    if first_path.size(1) != int(cfg.trajectory.horizon) + 1:
        raise ValueError(
            "Dataset path length must equal trajectory.horizon + 1; "
            f"got {first_path.size(1)} and {cfg.trajectory.horizon + 1}."
        )

    trajectory_encoder = TrajectoryEncoder(
        latent_dim=latent_dim,
        **cfg.trajectory_encoder,
    ).to(device)
    cost_model = TrajectoryCostModel(
        representation_dim=int(cfg.trajectory_encoder.representation_dim),
        dropout=float(cfg.trajectory_encoder.dropout),
    ).to(device)
    sigreg = None
    if float(cfg.loss.sigreg.weight) > 0:
        sigreg = SIGReg(**cfg.loss.sigreg.kwargs).to(device)

    trainable_parameters = list(trajectory_encoder.parameters()) + list(
        cost_model.parameters()
    )
    optimizer = torch.optim.AdamW(trainable_parameters, **cfg.optimizer)

    run_dir = Path(stablewm_cache_dir(sub_folder="checkpoints"), cfg.subdir)
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "ltc_config.yaml")
    wandb_run = init_wandb(cfg, run_dir)

    global_step = 0
    try:
        for epoch in range(int(cfg.epochs)):
            print(f"epoch={epoch + 1} train_start", flush=True)
            trajectory_encoder.train()
            cost_model.train()
            for batch_index, batch in enumerate(train_loader):
                output, cost_distributions = step_batch(
                    batch=batch,
                    lewm=lewm,
                    trajectory_encoder=trajectory_encoder,
                    cost_model=cost_model,
                    sigreg=sigreg,
                    cfg=cfg,
                    device=device,
                )
                optimizer.zero_grad(set_to_none=True)
                output["loss"].backward()
                grad_norm = None
                if cfg.grad_clip_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        trainable_parameters, float(cfg.grad_clip_norm)
                    )
                optimizer.step()
                global_step += 1

                if global_step % int(cfg.log_interval) == 0:
                    metrics = " ".join(
                        f"{name}={metric_float(value):.4f}"
                        for name, value in output.items()
                    )
                    print(f"epoch={epoch + 1} step={global_step} {metrics}", flush=True)

                if wandb_run is not None:
                    log_data = {
                        f"train/{name}": metric_float(value)
                        for name, value in output.items()
                    }
                    log_data["train/epoch"] = epoch + 1
                    log_data["train/lr"] = optimizer.param_groups[0]["lr"]
                    if grad_norm is not None:
                        log_data["train/grad_norm"] = float(grad_norm)
                    if global_step % int(cfg.wandb.cost_distribution_interval) == 0:
                        log_data.update(
                            cost_distribution_wandb_data(cost_distributions)
                        )
                    wandb_run.log(log_data, step=global_step)

                if cfg.max_train_batches is not None and batch_index + 1 >= int(cfg.max_train_batches):
                    break

            print(f"epoch={epoch + 1} validation_start", flush=True)
            validation = validate(
                loader=val_loader,
                lewm=lewm,
                trajectory_encoder=trajectory_encoder,
                cost_model=cost_model,
                sigreg=sigreg,
                cfg=cfg,
                device=device,
            )
            print(
                f"epoch={epoch + 1} validation "
                + " ".join(f"{name}={value:.4f}" for name, value in validation.items()),
                flush=True,
            )
            if wandb_run is not None:
                wandb_run.log(
                    {f"val/{name}": value for name, value in validation.items()}
                    | {"val/epoch": epoch + 1},
                    step=global_step,
                )

            print(f"epoch={epoch + 1} checkpoint_start", flush=True)
            ltc_path = save_ltc_checkpoint(
                run_dir=run_dir,
                epoch=epoch + 1,
                trajectory_encoder=trajectory_encoder,
                cost_model=cost_model,
                cfg=cfg,
            )
            print(
                f"epoch={epoch + 1} checkpoint_done ltc={ltc_path}",
                flush=True,
            )
            if wandb_run is not None and cfg.wandb.log_model:
                wandb_run.save(str(ltc_path), base_path=str(run_dir))
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    run()