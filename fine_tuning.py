"""Closed-loop phase-two fine-tuning for an experience-conditioned planner."""
from copy import deepcopy
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from latent_planner import (
    LatentPlannerRuntime,
    checkpoint_payload,
    flow_matching_loss,
    inverse_dynamics_loss,
    lewm_consistency_loss,
    load_ltc_components_for_evaluation,
    stablewm_cache_dir,
)
from module import SIGReg
from train_latent_planner import (
    encode_latents,
    experience_guidance_loss,
    freeze,
    init_wandb,
    make_loaders,
    metric_float,
)
from train_ltc import mismatched_goals, noised_path, preference_loss


def load_checkpoint_payload(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("Planner checkpoint must be a dictionary payload.")
    return payload


def load_ltc_for_finetuning(
    source_payload: dict[str, Any],
    cfg: DictConfig,
    latent_dim: int,
    device: torch.device,
):
    override = cfg.experience.get("ltc_checkpoint")
    ltc_payload = (
        load_checkpoint_payload(override)
        if override
        else source_payload.get("experience", {}).get("ltc")
    )
    if ltc_payload is None:
        raise ValueError(
            "Planner checkpoint contains no bundled LTC. Set "
            "experience.ltc_checkpoint only to override it."
        )
    encoder, cost_model = load_ltc_components_for_evaluation(
        ltc_payload, latent_dim=latent_dim, device=device
    )
    encoder.requires_grad_(True)
    cost_model.requires_grad_(True)
    return encoder, cost_model, ltc_payload


def updated_ltc_payload(
    base_payload: dict[str, Any],
    encoder: torch.nn.Module,
    cost_model: torch.nn.Module,
) -> dict[str, Any]:
    payload = deepcopy(base_payload)
    payload["trajectory_encoder"]["state_dict"] = encoder.state_dict()
    payload["cost_model"]["state_dict"] = cost_model.state_dict()
    return payload


def wandb_distribution(costs: torch.Tensor, prefix: str) -> dict[str, Any]:
    import wandb

    values = costs.detach().float().flatten().cpu()
    result: dict[str, Any] = {
        f"{prefix}/histogram": wandb.Histogram(values.numpy()),
        f"{prefix}/mean": float(values.mean()),
        f"{prefix}/std": float(values.std(unbiased=False)),
        f"{prefix}/min": float(values.min()),
        f"{prefix}/max": float(values.max()),
    }
    for quantile in (0.1, 0.5, 0.9):
        result[f"{prefix}/q{int(quantile * 100):02d}"] = float(
            torch.quantile(values, quantile)
        )
    return result


@torch.no_grad()
def collect(runtime, path, enc, cost, cfg, gen):
    """Generate a same-task FIFO with random candidates and matching rounds."""
    cap = int(cfg.collection.max_size)
    candidate_min = int(cfg.collection.candidates_min)
    candidate_max = int(cfg.collection.candidates_max)
    if cap <= 0 or candidate_min < 1 or candidate_max < candidate_min:
        raise ValueError("Invalid collection.max_size or candidate range")

    candidates = int(
        torch.randint(
            candidate_min,
            candidate_max + 1,
            (),
            device=path.device,
            generator=gen,
        )
    )
    rounds = (cap + candidates - 1) // candidates
    snap_r = int(torch.randint(0, rounds + 1, (), device=path.device, generator=gen))
    z0, goal, bank, snap = path[:, 0], path[:, -1], None, None
    state = (runtime.flow.training, enc.training, cost.training)
    runtime.flow.eval()
    enc.eval()
    cost.eval()
    try:
        for i in range(1, rounds + 1):
            feat, scores = memory_features(bank, goal, enc, cost)
            cand = runtime.sample_paths(
                z0,
                goal,
                horizon=int(cfg.planner.horizon),
                num_samples=candidates,
                flow_steps=int(cfg.collection.flow_steps),
                path_features=feat,
                path_costs=scores,
                generator=gen,
            ).detach()
            bank = cand if bank is None else torch.cat((bank, cand), dim=1)
            bank = bank[:, -cap:]
            if i == snap_r:
                snap = bank.clone()
    finally:
        runtime.flow.train(state[0])
        enc.train(state[1])
        cost.train(state[2])

    if bank is None:
        raise RuntimeError("Collection produced no experience paths")
    return (None if snap_r == 0 else snap, snap_r, bank, candidates, rounds)


def memory_features(bank, goal, enc, cost):
    if bank is None:
        return (None, None)
    (b, m, t, d) = bank.shape
    flat = bank.reshape(b * m, t, d)
    goals = goal[:, None].expand(-1, m, -1).reshape(b * m, d)
    feat = enc(flat, goals).reshape(b, m, -1)
    return (feat, cost(feat.flatten(0, 1)).reshape(b, m).detach())

def ltc_anchor(path, enc, cost, cfg):
    goal = path[:, -1]
    pos_feat = enc(path, goal)
    pos = cost(pos_feat)
    if path.size(0) > 1:
        mismatch = cost(enc(path, mismatched_goals(goal)))
        mismatch_loss = preference_loss(pos, mismatch, float(cfg.ltc.beta))
    else:
        mismatch = pos.detach()
        mismatch_loss = pos.new_zeros(())
    noisy = cost(enc(noised_path(path, float(cfg.ltc.noise_std)), goal))
    noise_loss = preference_loss(pos, noisy, float(cfg.ltc.beta))
    loss = (
        float(cfg.ltc.goal_mismatch_weight) * mismatch_loss
        + float(cfg.ltc.noise_weight) * noise_loss
    )
    return (
        loss,
        pos_feat,
        pos,
        {
            "positive_cost": pos.detach().mean(),
            "mismatch_cost": mismatch.detach().mean(),
            "noisy_cost": noisy.detach().mean(),
            "mismatch_margin": (mismatch - pos).detach().mean(),
            "noise_margin": (noisy - pos).detach().mean(),
            "goal_mismatch_loss": mismatch_loss.detach(),
            "noise_loss": noise_loss.detach(),
        },
    )


@torch.no_grad()
def expert_path_dynamic_labels(raw_memory, expert_path, cfg, gen):
    """Sample collected paths and label them by expert interior-path distance."""
    if raw_memory is None:
        return None, None
    b, m = raw_memory.shape[:2]
    count = min(int(cfg.dynamic_ltc.sample_size), m)
    indices = torch.rand(
        b, m, device=raw_memory.device, generator=gen
    ).argsort(dim=1)[:, :count]
    gather = indices[:, :, None, None].expand(
        -1, -1, raw_memory.size(2), raw_memory.size(3)
    )
    paths = raw_memory.gather(1, gather)
    # Candidate and expert paths share start and goal, so compare only the
    # learned intermediate trajectory rather than diluting error with endpoints.
    distance = (
        paths[:, :, 1:-1] - expert_path[:, None, 1:-1]
    ).square().mean(dim=(-1, -2))
    return paths, distance.detach()


def dynamic_ltc_loss(paths, expert_distances, goal, expert_cost, enc, cost, cfg):
    """Use expert-vs-path and better-path-vs-worse-path preference pairs."""
    if paths is None or expert_distances is None:
        zero = goal.new_zeros(())
        return zero, zero, zero, zero, zero
    b, n, t, d = paths.shape
    flat_paths = paths.reshape(b * n, t, d)
    goals = goal[:, None].expand(-1, n, -1).reshape(b * n, d)
    predicted = cost(enc(flat_paths, goals)).reshape(b, n)
    beta = float(cfg.ltc.beta)
    margin = float(cfg.dynamic_ltc.label_margin)

    # Expert has distance zero. Any collected path farther than the label
    # margin should receive a larger cost than that expert path.
    worse_than_expert = expert_distances > margin
    expert_pairs = F.softplus(
        -(predicted - expert_cost[:, None]) / beta
    )[worse_than_expert]
    expert_loss = (
        expert_pairs.mean() if expert_pairs.numel() else predicted.new_zeros(())
    )

    # Preserve relative quality among the collected, model-generated paths.
    better = expert_distances[:, :, None] + margin < expert_distances[:, None, :]
    path_pairs = F.softplus(
        -(predicted[:, None, :] - predicted[:, :, None]) / beta
    )[better]
    path_loss = path_pairs.mean() if path_pairs.numel() else predicted.new_zeros(())

    active = [
        loss for loss, valid in (
            (expert_loss, bool(worse_than_expert.any())),
            (path_loss, bool(better.any())),
        ) if valid
    ]
    loss = torch.stack(active).mean() if active else predicted.new_zeros(())
    return (
        loss,
        predicted.detach().mean(),
        expert_distances.mean(),
        expert_loss.detach(),
        path_loss.detach(),
    )

def train_step(batch, runtime, enc, cost, sigreg, cfg, device, gen):
    batch["action"] = torch.nan_to_num(batch["action"].to(device), 0.0)
    path = encode_latents(runtime.lewm, batch, device)
    horizon = int(cfg.planner.horizon)
    if path.size(1) != horizon + 1:
        raise ValueError("planner.horizon must match dataset latent path length")

    bank, snapshot_round, collected, candidates, rounds = collect(
        runtime, path, enc, cost, cfg, gen
    )
    if bank is not None:
        size = int(
            torch.randint(0, bank.size(1) + 1, (), device=device, generator=gen)
        )
        bank = None if size == 0 else bank[:, -size:]

    feat, scores = memory_features(bank, path[:, -1], enc, cost)
    flow = flow_matching_loss(runtime.flow, path, feat, scores, generator=gen)
    memory_loss, memory_metrics = experience_guidance_loss(
        flow=runtime.flow,
        z_path=path,
        path_features=feat,
        path_costs=scores,
        trajectory_encoder=enc,
        cost_model=cost,
        tau=float(cfg.loss.experience.tau),
    )
    inv, acts = inverse_dynamics_loss(
        runtime.inverse_dynamics, path, batch["action"][:, :horizon]
    )
    cons = (
        lewm_consistency_loss(
            runtime.lewm, path, acts, int(cfg.lewm_history_size)
        )
        if float(cfg.loss.consistency.weight)
        else path.new_zeros(())
    )

    static_ltc, pos_feat, positive_cost, metrics = ltc_anchor(path, enc, cost, cfg)
    dynamic_paths, expert_path_distance = expert_path_dynamic_labels(
        collected, path, cfg, gen
    )
    (
        dynamic_ltc,
        dynamic_cost,
        dynamic_distance,
        dynamic_expert_loss,
        dynamic_pairwise_loss,
    ) = dynamic_ltc_loss(
        dynamic_paths,
        expert_path_distance,
        path[:, -1],
        positive_cost,
        enc,
        cost,
        cfg,
    )
    ltc = static_ltc + float(cfg.dynamic_ltc.weight) * dynamic_ltc
    reg = (
        path.new_zeros(())
        if sigreg is None
        else sigreg(
            torch.cat(
                [pos_feat] + ([] if feat is None else [feat.flatten(0, 1)])
            ).unsqueeze(0)
        )
    )
    total = (
        float(cfg.loss.flow.weight) * flow
        + float(cfg.loss.inverse.weight) * inv
        + float(cfg.loss.consistency.weight) * cons
        + float(cfg.loss.experience.weight) * memory_loss
        + float(cfg.loss.ltc.weight) * ltc
        + float(cfg.loss.sigreg.weight) * reg
    )

    with torch.no_grad():
        _, real_costs = memory_features(collected, path[:, -1], enc, cost)
    assert real_costs is not None
    return {
        "loss": total,
        "flow_loss": flow.detach(),
        "inverse_loss": inv.detach(),
        "consistency_loss": cons.detach(),
        "experience_loss": memory_loss.detach(),
        **memory_metrics,
        "ltc_loss": ltc.detach(),
        "static_ltc_loss": static_ltc.detach(),
        "dynamic_ltc_loss": dynamic_ltc.detach(),
        "dynamic_cost": dynamic_cost,
        "dynamic_expert_path_distance": dynamic_distance,
        "dynamic_expert_vs_path_loss": dynamic_expert_loss,
        "dynamic_path_pairwise_loss": dynamic_pairwise_loss,
        "sigreg_loss": reg.detach(),
        "snapshot_round": path.new_tensor(snapshot_round),
        "memory_size": path.new_tensor(0 if bank is None else bank.size(1)),
        "collection_candidates": path.new_tensor(candidates),
        "collection_rounds": path.new_tensor(rounds),
        "real_experience_size": path.new_tensor(collected.size(1)),
        "real_experience_cost": real_costs.mean(),
        "real_experience_cost_std": real_costs.std(unbiased=False),
        "real_experience_cost_q10": torch.quantile(real_costs, 0.1),
        "real_experience_cost_q50": torch.quantile(real_costs, 0.5),
        "real_experience_cost_q90": torch.quantile(real_costs, 0.9),
        **metrics,
        "_real_experience_cost_samples": real_costs,
    }


@torch.no_grad()
def validate(loader, runtime, enc, cost, sigreg, cfg, device):
    """Evaluate the same closed-loop objective on held-out batches."""
    modes = (
        runtime.flow.training,
        runtime.inverse_dynamics.training,
        enc.training,
        cost.training,
    )
    runtime.flow.eval()
    runtime.inverse_dynamics.eval()
    enc.eval()
    cost.eval()
    generator = torch.Generator(device=device).manual_seed(int(cfg.seed) + 91_337)
    sums, count = {}, 0
    try:
        for index, batch in enumerate(loader):
            out = train_step(batch, runtime, enc, cost, sigreg, cfg, device, generator)
            out.pop("_real_experience_cost_samples", None)
            batch_size = batch["pixels"].size(0)
            count += batch_size
            for key, value in out.items():
                sums[key] = sums.get(key, 0.0) + metric_float(value) * batch_size
            if cfg.val_batches is not None and index + 1 >= int(cfg.val_batches):
                break
    finally:
        runtime.flow.train(modes[0])
        runtime.inverse_dynamics.train(modes[1])
        enc.train(modes[2])
        cost.train(modes[3])
    return {key: value / max(count, 1) for key, value in sums.items()}


def save_bundled_checkpoint(
    outdir, step, source_payload, runtime, enc, cost, base_ltc_payload, cfg
):
    """Save one reloadable planner+inverse-dynamics+LTC payload, excluding LeWM."""
    payload = checkpoint_payload(
        lewm_checkpoint=str(source_payload["lewm_checkpoint"]),
        action_block=runtime.action_block,
        flow=runtime.flow,
        inverse_dynamics=runtime.inverse_dynamics,
        cfg=OmegaConf.to_container(cfg, resolve=True),
    )
    payload["experience"] = {
        "phase": "closed_loop_fine_tuning",
        "ltc": updated_ltc_payload(base_ltc_payload, enc, cost),
    }
    step_path = outdir / f"{cfg.output_model_name}_step_{step}.pt"
    latest_path = outdir / f"{cfg.output_model_name}.pt"
    torch.save(payload, step_path)
    torch.save(payload, latest_path)
    print(f"step={step} checkpoint_done path={latest_path}", flush=True)
    return latest_path


@hydra.main(version_base=None, config_path="./config/train", config_name="fine_tuning")
def run(cfg: DictConfig):
    torch.manual_seed(int(cfg.seed))
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device).manual_seed(int(cfg.seed))

    train_loader, val_loader = make_loaders(cfg)
    source_payload = load_checkpoint_payload(cfg.planner_checkpoint)
    runtime = LatentPlannerRuntime.from_checkpoint(cfg.planner_checkpoint, device=device)
    runtime.lewm = freeze(runtime.lewm)
    if int(runtime.action_block) != int(cfg.planner.action_block):
        raise ValueError("action_block differs from phase-one checkpoint")

    first_path = encode_latents(runtime.lewm, next(iter(train_loader)), device)
    if first_path.size(1) != int(cfg.planner.horizon) + 1:
        raise ValueError("planner.horizon must match data")
    enc, cost, base_ltc_payload = load_ltc_for_finetuning(
        source_payload, cfg, first_path.size(-1), device
    )
    if enc.representation_dim != runtime.flow.path_feature_dim:
        raise ValueError("LTC representation_dim must equal flow.path_feature_dim")

    groups = [
        {"params": list(runtime.flow.parameters()) + list(runtime.inverse_dynamics.parameters()), "lr": float(cfg.optimizer.planner_lr)},
        {"params": enc.parameters(), "lr": float(cfg.optimizer.encoder_lr)},
        {"params": cost.parameters(), "lr": float(cfg.optimizer.cost_lr)},
    ]
    optimizer = torch.optim.AdamW(groups, weight_decay=float(cfg.optimizer.weight_decay))
    trainable_parameters = [p for group in groups for p in group["params"]]
    sigreg = SIGReg(**cfg.loss.sigreg.kwargs).to(device) if float(cfg.loss.sigreg.weight) else None

    outdir = Path(stablewm_cache_dir(sub_folder="checkpoints"), cfg.subdir)
    outdir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, outdir / "fine_tuning_config.yaml")
    wandb_run = init_wandb(cfg, outdir)
    step = 0

    def validate_and_save():
        print(f"step={step} validation_start", flush=True)
        validation = validate(val_loader, runtime, enc, cost, sigreg, cfg, device)
        print(
            f"step={step} validation "
            + " ".join(f"{key}={value:.4f}" for key, value in validation.items()),
            flush=True,
        )
        if wandb_run is not None:
            wandb_run.log({f"val/{key}": value for key, value in validation.items()}, step=step)
        save_bundled_checkpoint(
            outdir, step, source_payload, runtime, enc, cost, base_ltc_payload, cfg
        )

    try:
        for epoch in range(int(cfg.epochs)):
            print(f"epoch={epoch + 1} train_start", flush=True)
            runtime.flow.train()
            runtime.inverse_dynamics.train()
            enc.train()
            cost.train()
            for batch_index, batch in enumerate(train_loader):
                out = train_step(batch, runtime, enc, cost, sigreg, cfg, device, generator)
                real_cost_samples = out.pop("_real_experience_cost_samples")
                optimizer.zero_grad(set_to_none=True)
                out["loss"].backward()
                grad_norm = (
                    torch.nn.utils.clip_grad_norm_(trainable_parameters, float(cfg.grad_clip_norm))
                    if cfg.grad_clip_norm is not None else None
                )
                optimizer.step()
                step += 1

                if step % int(cfg.log_interval) == 0:
                    print(
                        f"epoch={epoch + 1} step={step} "
                        + " ".join(f"{key}={metric_float(value):.4f}" for key, value in out.items()),
                        flush=True,
                    )
                if wandb_run is not None:
                    log_data = {f"train/{key}": metric_float(value) for key, value in out.items()}
                    log_data |= {
                        "train/epoch": epoch + 1,
                        "train/lr_planner": optimizer.param_groups[0]["lr"],
                        "train/lr_encoder": optimizer.param_groups[1]["lr"],
                        "train/lr_cost": optimizer.param_groups[2]["lr"],
                    }
                    if grad_norm is not None:
                        log_data["train/grad_norm"] = float(grad_norm)
                    if step % int(cfg.wandb.real_experience_cost_distribution_interval) == 0:
                        log_data.update(wandb_distribution(real_cost_samples, "train/real_experience_cost_distribution"))
                    wandb_run.log(log_data, step=step)

                if step % int(cfg.validation_interval_steps) == 0:
                    validate_and_save()
                if cfg.max_train_batches is not None and batch_index + 1 >= int(cfg.max_train_batches):
                    break
            if step % int(cfg.validation_interval_steps) != 0:
                validate_and_save()
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == '__main__':
    run()
