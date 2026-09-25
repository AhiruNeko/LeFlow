import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf, open_dict

from latent_planner import (
    InverseDynamics,
    LatentPathFlow,
    LatentPlannerRuntime,
    checkpoint_payload,
    load_lewm,
    stablewm_cache_dir,
)
from latent_trajectory_cost import TrajectoryCostModel, TrajectoryEncoder
from module import SIGReg
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


def _ltc_component_payloads(
    trajectory_encoder: TrajectoryEncoder,
    cost_model: TrajectoryCostModel,
    *,
    lewm_checkpoint: str,
    cfg: DictConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
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
        "output_activation": cost_model.output_activation,
    }
    common = {
        "format": "latent_trajectory_cost_component_v1",
        "lewm_checkpoint": lewm_checkpoint,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    return (
        common | {
            "component": "trajectory_encoder",
            "architecture": encoder_arch,
            "state_dict": trajectory_encoder.state_dict(),
        },
        common | {
            "component": "trajectory_cost",
            "architecture": cost_arch,
            "state_dict": cost_model.state_dict(),
        },
    )


def _ltc_components_from_payload(
    payload: dict[str, Any], *, latent_dim: int, device: torch.device
) -> tuple[TrajectoryEncoder, TrajectoryCostModel]:
    """Instantiate LTC modules from an LTC bundle or unified planner checkpoint."""
    if payload.get("format") == "latent_trajectory_cost_bundle_v1":
        encoder_payload = payload["trajectory_encoder"]
        cost_payload = payload["cost_model"]
    else:
        experience = payload.get("experience", {})
        encoder_payload = experience.get("trajectory_encoder")
        cost_payload = experience.get("cost_model")
    if not isinstance(encoder_payload, dict) or not isinstance(cost_payload, dict):
        raise ValueError("checkpoint does not contain bundled trajectory encoder and cost model")
    if encoder_payload.get("component") != "trajectory_encoder":
        raise ValueError("bundled trajectory encoder payload is invalid")
    if cost_payload.get("component") != "trajectory_cost":
        raise ValueError("bundled cost-model payload is invalid")
    encoder = TrajectoryEncoder.from_checkpoint_architecture(
        latent_dim=latent_dim, architecture=dict(encoder_payload["architecture"])
    ).to(device)
    cost_model = TrajectoryCostModel.from_checkpoint_architecture(
        dict(cost_payload["architecture"])
    ).to(device)
    encoder.load_state_dict(encoder_payload["state_dict"], strict=True)
    cost_model.load_state_dict(cost_payload["state_dict"], strict=True)
    return encoder, cost_model


def load_ltc_components(
    cfg: DictConfig, latent_dim: int, device: torch.device
) -> tuple[TrajectoryEncoder, TrajectoryCostModel]:
    """Load both LTC modules from a single bundled pretraining checkpoint."""
    return _ltc_components_from_payload(
        _checkpoint_load(cfg.experience.checkpoint), latent_dim=latent_dim, device=device
    )

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


def preference_loss(
    positive_cost: torch.Tensor, negative_cost: torch.Tensor, beta: float
) -> torch.Tensor:
    """Pairwise preference loss: lower cost is better."""
    if beta <= 0:
        raise ValueError("ltc.beta must be positive")
    return F.softplus(-(negative_cost - positive_cost) / beta).mean()


def paired_flow_losses(
    flow: LatentPathFlow,
    z_path: torch.Tensor,
    path_features: torch.Tensor | None,
    path_costs: torch.Tensor | None,
    *,
    use_margin: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compare no-memory and memory-conditioned fields for identical x_t and t.

    The relative term can only improve the memory branch because the no-memory
    error is detached.  It is therefore a direct signal that experience must
    reduce flow-matching error rather than merely be present in the network.
    """
    z_start, z_goal = z_path[:, 0], z_path[:, -1]
    target = z_path[:, 1:-1]
    noise = torch.randn(
        target.shape, device=target.device, dtype=target.dtype, generator=generator
    )
    t = torch.rand(
        z_path.size(0), device=z_path.device, dtype=z_path.dtype, generator=generator
    )
    x_t = (1 - t[:, None, None]) * noise + t[:, None, None] * target
    velocity_target = target - noise

    empty_prediction = flow(x_t, t, z_start, z_goal)
    empty_error = (empty_prediction - velocity_target).square().mean(dim=(1, 2))
    empty_loss = empty_error.mean()
    if path_features is None:
        zero = empty_loss.new_zeros(())
        return empty_loss, zero, zero, empty_loss.detach(), zero

    memory_prediction = flow(
        x_t,
        t,
        z_start,
        z_goal,
        path_features=path_features,
        path_costs=path_costs,
    )
    memory_error = (memory_prediction - velocity_target).square().mean(dim=(1, 2))
    memory_loss = memory_error.mean()
    use_loss = F.relu(memory_error - empty_error.detach() + use_margin).mean()
    return empty_loss, memory_loss, use_loss, empty_error.detach().mean(), memory_error.detach().mean()


def make_synthetic_negative(path: torch.Tensor, cfg: DictConfig) -> torch.Tensor:
    """One same-task noisy path per expert path for memory and LTC supervision."""
    all_after_start = noised_path_all_after_start(path, float(cfg.synthetic.noise_std))
    intermediate_only = noised_path_intermediate(path, float(cfg.synthetic.noise_std))
    choose_all = torch.rand(path.size(0), device=path.device) < float(
        cfg.synthetic.all_after_start_probability
    )
    return torch.where(choose_all[:, None, None], all_after_start, intermediate_only)


def sigreg_loss(
    sigreg: SIGReg | None, *features: torch.Tensor
) -> torch.Tensor:
    if sigreg is None:
        return features[0].new_zeros(())
    return sigreg(torch.stack(features, dim=0))


def synthetic_losses(
    z_path: torch.Tensor,
    flow: LatentPathFlow,
    encoder: TrajectoryEncoder,
    cost_model: TrajectoryCostModel,
    sigreg: SIGReg | None,
    cfg: DictConfig,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """One synthetic experience and one LTC pair for every planner update."""
    goal = z_path[:, -1]
    negative_path = make_synthetic_negative(z_path, cfg)
    positive_feature = encoder(z_path, goal)
    negative_feature = encoder(negative_path, goal)
    positive_cost = cost_model(positive_feature)
    negative_cost = cost_model(negative_feature)

    cost_pair = preference_loss(positive_cost, negative_cost, float(cfg.ltc.beta))
    regularizer = sigreg_loss(sigreg, positive_feature, negative_feature)
    cost_loss = cost_pair + float(cfg.loss.sigreg.weight) * regularizer

    empty, memory, use, empty_error, memory_error = paired_flow_losses(
        flow,
        z_path,
        negative_feature[:, None],
        negative_cost.detach()[:, None],
        use_margin=float(cfg.loss.use.margin),
        generator=generator,
    )
    planner_loss = (
        float(cfg.loss.flow.empty_weight) * empty
        + float(cfg.loss.flow.memory_weight) * memory
        + float(cfg.loss.use.weight) * use
    )
    metrics = {
        "planner_loss": planner_loss.detach(),
        "cost_loss": cost_loss.detach(),
        "empty_flow_loss": empty.detach(),
        "memory_flow_loss": memory.detach(),
        "use_loss": use.detach(),
        "empty_flow_error": empty_error,
        "memory_flow_error": memory_error,
        "positive_cost": positive_cost.detach().mean(),
        "negative_cost": negative_cost.detach().mean(),
        "cost_margin": (negative_cost - positive_cost).detach().mean(),
        "pairwise_cost_loss": cost_pair.detach(),
        "sigreg_loss": regularizer.detach(),
        "experience_size": z_path.new_tensor(1),
    }
    return planner_loss, cost_loss, metrics


@dataclass
class RealExperience:
    """Detached same-task model paths and their frozen LeWM rollout distances."""

    expert_path: torch.Tensor
    paths: torch.Tensor
    rollout_distance: torch.Tensor
    rounds: int
    samples_per_round: int


@torch.no_grad()
def collect_real_experience(
    runtime: LatentPlannerRuntime,
    z_path: torch.Tensor,
    encoder: TrajectoryEncoder,
    cost_model: TrajectoryCostModel,
    cfg: DictConfig,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Collect FIFO candidates with random sampling budget and real LeWM scores."""
    candidates = int(
        cfg.collection.candidates_choices[
            int(torch.randint(len(cfg.collection.candidates_choices), (), device=z_path.device, generator=generator))
        ]
    )
    rounds = int(
        cfg.collection.rounds_choices[
            int(torch.randint(len(cfg.collection.rounds_choices), (), device=z_path.device, generator=generator))
        ]
    )
    capacity = int(cfg.collection.max_paths_per_task)
    z_start, goal = z_path[:, 0], z_path[:, -1]
    raw_paths = distances = None
    module_state = (runtime.flow.training, encoder.training, cost_model.training)
    runtime.flow.eval()
    encoder.eval()
    cost_model.eval()
    try:
        for _ in range(rounds):
            if raw_paths is None:
                features = costs = None
            else:
                batch, count, steps, dim = raw_paths.shape
                flat_paths = raw_paths.reshape(batch * count, steps, dim)
                flat_goals = goal[:, None].expand(-1, count, -1).reshape(batch * count, dim)
                features = encoder(flat_paths, flat_goals).reshape(batch, count, -1)
                costs = cost_model(features.flatten(0, 1)).reshape(batch, count)
            candidates_path = runtime.sample_paths(
                z_start,
                goal,
                horizon=int(cfg.planner.horizon),
                num_samples=candidates,
                flow_steps=int(cfg.collection.flow_steps),
                path_features=features,
                path_costs=costs,
                generator=generator,
            )
            actions = runtime.decode_actions(candidates_path)
            final_latent = runtime.rollout_final_latent(
                z_start, actions, history_size=int(cfg.lewm_history_size)
            )
            candidate_distance = (final_latent - goal[:, None]).square().mean(dim=-1)
            raw_paths = (
                candidates_path
                if raw_paths is None
                else torch.cat((raw_paths, candidates_path), dim=1)
            )
            distances = (
                candidate_distance
                if distances is None
                else torch.cat((distances, candidate_distance), dim=1)
            )
            if capacity:
                raw_paths = raw_paths[:, -capacity:]
                distances = distances[:, -capacity:]
    finally:
        runtime.flow.train(module_state[0])
        encoder.train(module_state[1])
        cost_model.train(module_state[2])
    assert raw_paths is not None and distances is not None
    return raw_paths.detach(), distances.detach(), rounds, candidates


def sample_real_memory(
    paths: torch.Tensor,
    goal: torch.Tensor,
    encoder: TrajectoryEncoder,
    cost_model: TrajectoryCostModel,
    cfg: DictConfig,
    generator: torch.Generator,
) -> tuple[torch.Tensor | None, torch.Tensor | None, int]:
    """Choose a random FIFO subset; feature gradients flow to the encoder only."""
    batch, available, steps, dim = paths.shape
    allowed = [int(size) for size in cfg.real_replay.memory_size_choices if int(size) <= available]
    if not allowed:
        return None, None, 0
    count = allowed[
        int(torch.randint(
            len(allowed), (), device=paths.device, generator=generator
        ))
    ]
    if count == 0:
        return None, None, 0
    indices = torch.rand(batch, available, device=paths.device, generator=generator).argsort(dim=1)[:, :count]
    gather = indices[:, :, None, None].expand(-1, -1, steps, dim)
    selected = paths.gather(1, gather)
    flat = selected.reshape(batch * count, steps, dim)
    goals = goal[:, None].expand(-1, count, -1).reshape(batch * count, dim)
    features = encoder(flat, goals).reshape(batch, count, -1)
    costs = cost_model(features.flatten(0, 1)).reshape(batch, count)
    return features, costs.detach(), count


def real_losses(
    record: RealExperience,
    flow: LatentPathFlow,
    encoder: TrajectoryEncoder,
    cost_model: TrajectoryCostModel,
    sigreg: SIGReg | None,
    cfg: DictConfig,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Train on true generated paths, ranking cost by detached LeWM rollout quality."""
    z_path = record.expert_path.to(device, non_blocking=True)
    paths = record.paths.to(device, non_blocking=True)
    rollout_distance = record.rollout_distance.to(device, non_blocking=True)
    goal = z_path[:, -1]
    features, costs, size = sample_real_memory(
        paths, goal, encoder, cost_model, cfg, generator
    )
    empty, memory, use, empty_error, memory_error = paired_flow_losses(
        flow,
        z_path,
        features,
        costs,
        use_margin=float(cfg.loss.use.margin),
        generator=generator,
    )
    planner_loss = (
        float(cfg.loss.flow.empty_weight) * empty
        + float(cfg.loss.flow.memory_weight) * memory
        + float(cfg.loss.use.weight) * use
    )

    best = rollout_distance.argmin(dim=1)
    worst = rollout_distance.argmax(dim=1)
    batch_index = torch.arange(z_path.size(0), device=device)
    positive_path = paths[batch_index, best]
    negative_path = paths[batch_index, worst]
    positive_feature = encoder(positive_path, goal)
    negative_feature = encoder(negative_path, goal)
    positive_cost = cost_model(positive_feature)
    negative_cost = cost_model(negative_feature)
    cost_pair = preference_loss(positive_cost, negative_cost, float(cfg.ltc.beta))
    regularizer = sigreg_loss(sigreg, positive_feature, negative_feature)
    cost_loss = cost_pair + float(cfg.loss.sigreg.weight) * regularizer
    metrics = {
        "planner_loss": planner_loss.detach(),
        "cost_loss": cost_loss.detach(),
        "empty_flow_loss": empty.detach(),
        "memory_flow_loss": memory.detach(),
        "use_loss": use.detach(),
        "empty_flow_error": empty_error,
        "memory_flow_error": memory_error,
        "positive_cost": positive_cost.detach().mean(),
        "negative_cost": negative_cost.detach().mean(),
        "cost_margin": (negative_cost - positive_cost).detach().mean(),
        "pairwise_cost_loss": cost_pair.detach(),
        "sigreg_loss": regularizer.detach(),
        "experience_size": z_path.new_tensor(size),
        "rollout_distance_best": rollout_distance.min(dim=1).values.detach().mean(),
        "rollout_distance_worst": rollout_distance.max(dim=1).values.detach().mean(),
        "collection_rounds": z_path.new_tensor(record.rounds),
        "collection_candidates": z_path.new_tensor(record.samples_per_round),
    }
    return planner_loss, cost_loss, metrics


def optimize_losses(
    planner_loss: torch.Tensor,
    cost_loss: torch.Tensor,
    planner_optimizer: torch.optim.Optimizer,
    encoder_optimizer: torch.optim.Optimizer,
    cost_optimizer: torch.optim.Optimizer,
    flow: LatentPathFlow,
    encoder: TrajectoryEncoder,
    cost_model: TrajectoryCostModel,
    grad_clip_norm: float | None,
) -> dict[str, torch.Tensor]:
    """Route gradients: flow ← planner, cost ← cost, encoder ← both."""
    for optimizer in (planner_optimizer, encoder_optimizer, cost_optimizer):
        optimizer.zero_grad(set_to_none=True)
    planner_loss.backward(retain_graph=True)
    cost_loss.backward()
    metrics: dict[str, torch.Tensor] = {}
    if grad_clip_norm is not None:
        metrics["planner_grad_norm"] = torch.nn.utils.clip_grad_norm_(
            flow.parameters(), float(grad_clip_norm)
        ).detach()
        metrics["encoder_grad_norm"] = torch.nn.utils.clip_grad_norm_(
            encoder.parameters(), float(grad_clip_norm)
        ).detach()
        metrics["cost_grad_norm"] = torch.nn.utils.clip_grad_norm_(
            cost_model.parameters(), float(grad_clip_norm)
        ).detach()
    for optimizer in (planner_optimizer, encoder_optimizer, cost_optimizer):
        optimizer.step()
    return metrics


def take_batch(iterator, loader):
    """Take one batch, restarting only for the optional bootstrap warm-up."""
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def cycle_batch_counts(total_batches: int, cycles_per_epoch: int) -> list[int]:
    """Split one complete loader pass into non-empty in-epoch cycle chunks."""
    if total_batches < 1:
        raise ValueError("the training loader must contain at least one batch")
    if cycles_per_epoch < 1 or cycles_per_epoch > total_batches:
        raise ValueError(
            "training.cycles_per_epoch must be in [1, len(train_loader)]; "
            f"got {cycles_per_epoch} for {total_batches} loader batches"
        )
    base, remainder = divmod(total_batches, cycles_per_epoch)
    return [base + int(index < remainder) for index in range(cycles_per_epoch)]


def slice_batch(batch: dict[str, torch.Tensor], size: int) -> dict[str, torch.Tensor]:
    return {
        key: value[:size] if torch.is_tensor(value) and value.ndim and value.size(0) >= size else value
        for key, value in batch.items()
    }


def log_metrics(
    *,
    phase: str,
    epoch: int,
    cycle: int,
    step: int,
    metrics: dict[str, torch.Tensor],
    wandb_run,
) -> None:
    text = " ".join(f"{key}={metric_float(value):.4f}" for key, value in metrics.items())
    print(
        f"epoch={epoch} cycle={cycle} step={step} phase={phase} {text}",
        flush=True,
    )
    if wandb_run is not None:
        wandb_run.log(
            {f"{phase}/{key}": metric_float(value) for key, value in metrics.items()}
            | {"train/epoch": epoch, "train/cycle": cycle, "train/step": step},
            step=step,
        )


def load_frozen_inverse_dynamics(
    checkpoint: str | Path, device: torch.device, expected_action_block: int
) -> InverseDynamics:
    """Extract the original LeFlow inverse dynamics from its bundled checkpoint."""
    payload = _checkpoint_load(checkpoint)
    if "arch" not in payload or "inverse_dynamics_state_dict" not in payload:
        raise ValueError("inverse_dynamics_checkpoint must be a LeFlow planner checkpoint")
    checkpoint_action_block = int(payload.get("action_block", expected_action_block))
    if checkpoint_action_block != expected_action_block:
        raise ValueError(
            f"inverse-dynamics action_block={checkpoint_action_block} does not match "
            f"planner.action_block={expected_action_block}"
        )
    inverse = InverseDynamics(**payload["arch"]["inverse_dynamics"]).to(device)
    inverse.load_state_dict(payload["inverse_dynamics_state_dict"], strict=True)
    return freeze(inverse)


def save_unified_checkpoint(
    *,
    run_dir: Path,
    name: str,
    epoch: int,
    source_inverse_checkpoint: str,
    lewm_checkpoint: str,
    flow: LatentPathFlow,
    inverse_dynamics: InverseDynamics,
    encoder: TrajectoryEncoder,
    cost_model: TrajectoryCostModel,
    cfg: DictConfig,
) -> None:
    payload = checkpoint_payload(
        lewm_checkpoint=lewm_checkpoint,
        action_block=int(cfg.planner.action_block),
        flow=flow,
        inverse_dynamics=inverse_dynamics,
        cfg=OmegaConf.to_container(cfg, resolve=True),
    )
    encoder_payload, cost_payload = _ltc_component_payloads(
        encoder,
        cost_model,
        lewm_checkpoint=lewm_checkpoint,
        cfg=cfg,
    )
    # One deployment checkpoint: flow, frozen original IDM, and both LTC modules.
    payload["experience"] = {
        "training_mode": "unified_synthetic_and_real_experience",
        "inverse_dynamics_checkpoint": source_inverse_checkpoint,
        "inverse_dynamics_frozen": True,
        "trajectory_encoder": encoder_payload,
        "cost_model": cost_payload,
    }
    torch.save(payload, run_dir / f"{name}_epoch_{epoch}.pt")
    torch.save(payload, run_dir / f"{name}.pt")


@hydra.main(version_base=None, config_path="./config/train", config_name="latent_planner")
def run(cfg: DictConfig) -> None:
    torch.manual_seed(int(cfg.seed))
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(int(cfg.seed))
    train_loader, _ = make_loaders(cfg)
    lewm = freeze(load_lewm(cfg.lewm_checkpoint).to(device))
    first_batch = next(iter(train_loader))
    first_path = encode_latents(lewm, first_batch, device)
    latent_dim = first_path.size(-1)
    if first_path.size(1) != int(cfg.planner.horizon) + 1:
        raise ValueError("planner.horizon must match the dataset latent path length")

    flow = LatentPathFlow(
        latent_dim=latent_dim,
        max_horizon=int(cfg.planner.max_horizon),
        **cfg.flow,
    ).to(device)
    inverse_dynamics = load_frozen_inverse_dynamics(
        cfg.inverse_dynamics_checkpoint,
        device,
        int(cfg.planner.action_block),
    )
    encoder, cost_model = load_ltc_components(cfg, latent_dim, device)
    if encoder.representation_dim != flow.path_feature_dim:
        raise ValueError("LTC representation_dim must equal flow.path_feature_dim")
    runtime = LatentPlannerRuntime(lewm, flow, inverse_dynamics, int(cfg.planner.action_block)).to(device)

    planner_optimizer = torch.optim.AdamW(flow.parameters(), **cfg.optimizer.planner)
    encoder_optimizer = torch.optim.AdamW(encoder.parameters(), **cfg.optimizer.encoder)
    cost_optimizer = torch.optim.AdamW(cost_model.parameters(), **cfg.optimizer.cost)
    sigreg = SIGReg(**cfg.loss.sigreg.kwargs).to(device) if float(cfg.loss.sigreg.weight) else None

    run_dir = Path(stablewm_cache_dir(sub_folder="checkpoints"), cfg.subdir)
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "latent_planner_config.yaml")
    wandb_run = init_wandb(cfg, run_dir)
    replay: deque[RealExperience] = deque(maxlen=int(cfg.real_replay.cache_max_banks))
    global_step = 0

    try:
        # Bootstrap is deliberately step-based because it is only a short
        # warm-up before the first complete epoch.
        bootstrap_iterator = iter(train_loader)
        for bootstrap_step in range(int(cfg.training.bootstrap_synthetic_updates)):
            batch, bootstrap_iterator = take_batch(bootstrap_iterator, train_loader)
            z_path = encode_latents(lewm, batch, device)
            flow.train(); encoder.train(); cost_model.train(); inverse_dynamics.eval()
            planner_loss, cost_loss, metrics = synthetic_losses(
                z_path, flow, encoder, cost_model, sigreg, cfg, generator
            )
            metrics |= optimize_losses(
                planner_loss, cost_loss, planner_optimizer, encoder_optimizer,
                cost_optimizer, flow, encoder, cost_model, cfg.grad_clip_norm,
            )
            global_step += 1
            if global_step % int(cfg.log_interval) == 0:
                log_metrics(
                    phase="bootstrap", epoch=0, cycle=0, step=global_step,
                    metrics=metrics, wandb_run=wandb_run,
                )

        batches_per_epoch = len(train_loader)
        cycle_sizes = cycle_batch_counts(
            batches_per_epoch, int(cfg.training.cycles_per_epoch)
        )
        print(
            f"epoch_schedule batches_per_epoch={batches_per_epoch} "
            f"cycles_per_epoch={len(cycle_sizes)} "
            f"synthetic_batches_per_cycle={cycle_sizes}",
            flush=True,
        )

        for epoch in range(1, int(cfg.training.epochs) + 1):
            print(f"epoch={epoch} train_start", flush=True)
            epoch_iterator = iter(train_loader)

            for cycle, synthetic_batches in enumerate(cycle_sizes, start=1):
                # Stage 1: consume this exact partition of the epoch. The final
                # task batch also seeds collection, so no additional loader batch
                # is consumed outside the full epoch traversal.
                collection_source = None
                for _ in range(synthetic_batches):
                    batch = next(epoch_iterator)
                    collection_source = batch
                    z_path = encode_latents(lewm, batch, device)
                    flow.train(); encoder.train(); cost_model.train(); inverse_dynamics.eval()
                    planner_loss, cost_loss, metrics = synthetic_losses(
                        z_path, flow, encoder, cost_model, sigreg, cfg, generator
                    )
                    metrics |= optimize_losses(
                        planner_loss, cost_loss, planner_optimizer, encoder_optimizer,
                        cost_optimizer, flow, encoder, cost_model, cfg.grad_clip_norm,
                    )
                    global_step += 1
                    if global_step % int(cfg.log_interval) == 0:
                        log_metrics(
                            phase="synthetic", epoch=epoch, cycle=cycle,
                            step=global_step, metrics=metrics, wandb_run=wandb_run,
                        )

                if collection_source is None:
                    raise RuntimeError("in-epoch cycle has no synthetic batch")

                # Stage 2: collect real candidates for tasks already consumed by
                # this epoch's synthetic partition; frozen LeWM/IDM supply labels.
                collection_batch = slice_batch(
                    collection_source, int(cfg.collection.task_batch_size)
                )
                collection_path = encode_latents(lewm, collection_batch, device)
                raw_paths, rollout_distance, rounds, candidates = collect_real_experience(
                    runtime, collection_path, encoder, cost_model, cfg, generator
                )
                replay.append(
                    RealExperience(
                        expert_path=collection_path.detach().cpu(),
                        paths=raw_paths.detach().cpu(),
                        rollout_distance=rollout_distance.detach().cpu(),
                        rounds=rounds,
                        samples_per_round=candidates,
                    )
                )
                cached_tasks = sum(record.expert_path.size(0) for record in replay)
                collection_metrics = {
                    "collection_paths_per_task": collection_path.new_tensor(raw_paths.size(1)),
                    "collection_rounds": collection_path.new_tensor(rounds),
                    "collection_candidates": collection_path.new_tensor(candidates),
                    "collection_best_rollout_distance": rollout_distance.min(dim=1).values.mean(),
                    "collection_cache_tasks": collection_path.new_tensor(cached_tasks),
                    "collection_cache_banks": collection_path.new_tensor(len(replay)),
                }
                log_metrics(
                    phase="collection", epoch=epoch, cycle=cycle,
                    step=global_step, metrics=collection_metrics, wandb_run=wandb_run,
                )

                # Stage 3: replay real paths using a random historical bank and a
                # random FIFO length; no extra original-dataset batch is consumed.
                for _ in range(int(cfg.training.real_updates_per_cycle)):
                    index = int(torch.randint(len(replay), (), device=device, generator=generator))
                    record = replay[index]
                    flow.train(); encoder.train(); cost_model.train(); inverse_dynamics.eval()
                    planner_loss, cost_loss, metrics = real_losses(
                        record, flow, encoder, cost_model, sigreg, cfg, generator, device
                    )
                    metrics |= optimize_losses(
                        planner_loss, cost_loss, planner_optimizer, encoder_optimizer,
                        cost_optimizer, flow, encoder, cost_model, cfg.grad_clip_norm,
                    )
                    global_step += 1
                    if global_step % int(cfg.log_interval) == 0:
                        log_metrics(
                            phase="real", epoch=epoch, cycle=cycle,
                            step=global_step, metrics=metrics, wandb_run=wandb_run,
                        )

            if epoch % int(cfg.training.checkpoint_every_epochs) == 0:
                save_unified_checkpoint(
                    run_dir=run_dir,
                    name=str(cfg.output_model_name),
                    epoch=epoch,
                    source_inverse_checkpoint=str(cfg.inverse_dynamics_checkpoint),
                    lewm_checkpoint=str(cfg.lewm_checkpoint),
                    flow=flow,
                    inverse_dynamics=inverse_dynamics,
                    encoder=encoder,
                    cost_model=cost_model,
                    cfg=cfg,
                )
                print(f"epoch={epoch} checkpoint_done run_dir={run_dir}", flush=True)
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    run()