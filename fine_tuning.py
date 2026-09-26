"""Closed-loop phase-two fine-tuning for an experience-conditioned planner."""
from pathlib import Path
import hydra, torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from latent_planner import LatentPlannerRuntime, checkpoint_payload, flow_matching_loss, inverse_dynamics_loss, lewm_consistency_loss, stablewm_cache_dir
from module import SIGReg
from train_latent_planner import encode_latents, freeze, init_wandb, load_ltc_components, make_loaders, metric_float, save_finetuned_ltc_components
from train_ltc import mismatched_goals, noised_path, preference_loss

@torch.no_grad()
def collect(runtime, path, enc, cost, cfg, gen):
    """Generate same-task raw paths into FIFO; return a random intermediate bank."""
    (r, n, cap) = (int(cfg.collection.rounds), int(cfg.collection.samples_per_round), int(cfg.collection.max_size))
    snap_r = int(torch.randint(0, r + 1, (), device=path.device, generator=gen))
    (z0, goal, bank, snap) = (path[:, 0], path[:, -1], None, None)
    state = (runtime.flow.training, enc.training, cost.training)
    runtime.flow.eval()
    enc.eval()
    cost.eval()
    try:
        for i in range(1, r + 1):
            if bank is None:
                feat = scores = None
            else:
                (b, m, t, d) = bank.shape
                flat = bank.reshape(b * m, t, d)
                goals = goal[:, None].expand(-1, m, -1).reshape(b * m, d)
                feat = enc(flat, goals).reshape(b, m, -1)
                scores = cost(feat.flatten(0, 1)).reshape(b, m)
            cand = runtime.sample_paths(z0, goal, horizon=int(cfg.planner.horizon), num_samples=n, flow_steps=int(cfg.collection.flow_steps), path_features=feat, path_costs=scores, generator=gen).detach()
            bank = cand if bank is None else torch.cat((bank, cand), 1)
            bank = bank[:, -cap:] if cap else None
            if i == snap_r:
                snap = None if bank is None else bank.clone()
    finally:
        runtime.flow.train(state[0])
        enc.train(state[1])
        cost.train(state[2])
    return (None if snap_r == 0 else snap, snap_r, bank)

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
    loss = float(cfg.ltc.goal_mismatch_weight) * mismatch_loss + float(cfg.ltc.noise_weight) * noise_loss
    return (loss, pos_feat, {'positive_cost': pos.detach().mean(), 'mismatch_cost': mismatch.detach().mean(), 'noisy_cost': noisy.detach().mean(), 'mismatch_margin': (mismatch - pos).detach().mean(), 'noise_margin': (noisy - pos).detach().mean(), 'goal_mismatch_loss': mismatch_loss.detach(), 'noise_loss': noise_loss.detach()})

@torch.no_grad()
def rollout_dynamic_labels(runtime, raw_memory, z_start, z_goal, cfg, gen):
    """Sample collected paths and produce detached LeWM rollout quality labels."""
    if raw_memory is None:
        return None, None
    b, m = raw_memory.shape[:2]
    count = min(int(cfg.dynamic_ltc.sample_size), m)
    indices = torch.rand(b, m, device=raw_memory.device, generator=gen).argsort(dim=1)[:, :count]
    gather = indices[:, :, None, None].expand(-1, -1, raw_memory.size(2), raw_memory.size(3))
    paths = raw_memory.gather(1, gather)
    actions = runtime.decode_actions(paths)
    final_latent = runtime.rollout_final_latent(z_start, actions, int(cfg.lewm_history_size))
    distance = (final_latent - z_goal[:, None]).square().mean(dim=-1)
    return paths, distance.detach()

def dynamic_ltc_loss(paths, distances, goal, enc, cost, cfg):
    """Rank sampled model paths using detached rollout terminal distances."""
    if paths is None or distances is None:
        return goal.new_zeros(()), goal.new_zeros(()), goal.new_zeros(())
    b, n, t, d = paths.shape
    flat_paths = paths.reshape(b * n, t, d)
    goals = goal[:, None].expand(-1, n, -1).reshape(b * n, d)
    predicted = cost(enc(flat_paths, goals)).reshape(b, n)
    better = distances[:, :, None] + float(cfg.dynamic_ltc.label_margin) < distances[:, None, :]
    if not bool(better.any()):
        return predicted.new_zeros(()), predicted.detach().mean(), distances.mean()
    loss = F.softplus(-(predicted[:, None, :] - predicted[:, :, None]) / float(cfg.ltc.beta))[better].mean()
    return loss, predicted.detach().mean(), distances.mean()

def train_step(batch, runtime, enc, cost, sigreg, cfg, device, gen):
    batch['action'] = torch.nan_to_num(batch['action'].to(device), 0.0)
    path = encode_latents(runtime.lewm, batch, device)
    h = int(cfg.planner.horizon)
    if path.size(1) != h + 1:
        raise ValueError('planner.horizon must match dataset latent path length')
    (bank, snap, collected) = collect(runtime, path, enc, cost, cfg, gen)
    if bank is not None:
        size = int(torch.randint(0, bank.size(1) + 1, (), device=device, generator=gen))
        bank = None if size == 0 else bank[:, -size:]
    (feat, scores) = memory_features(bank, path[:, -1], enc, cost)
    flow = flow_matching_loss(runtime.flow, path, feat, scores, generator=gen)
    (inv, acts) = inverse_dynamics_loss(runtime.inverse_dynamics, path, batch['action'][:, :h])
    cons = lewm_consistency_loss(runtime.lewm, path, acts, int(cfg.lewm_history_size)) if float(cfg.loss.consistency.weight) else path.new_zeros(())
    (static_ltc, pos_feat, metrics) = ltc_anchor(path, enc, cost, cfg)
    dynamic_paths, rollout_distance = rollout_dynamic_labels(runtime, collected, path[:, 0], path[:, -1], cfg, gen)
    dynamic_ltc, dynamic_cost, dynamic_distance = dynamic_ltc_loss(dynamic_paths, rollout_distance, path[:, -1], enc, cost, cfg)
    ltc = static_ltc + float(cfg.dynamic_ltc.weight) * dynamic_ltc
    reg = path.new_zeros(()) if sigreg is None else sigreg(torch.cat([pos_feat] + ([] if feat is None else [feat.flatten(0, 1)])).unsqueeze(0))
    total = float(cfg.loss.flow.weight) * flow + float(cfg.loss.inverse.weight) * inv + float(cfg.loss.consistency.weight) * cons + float(cfg.loss.ltc.weight) * ltc + float(cfg.loss.sigreg.weight) * reg
    return {'loss': total, 'flow_loss': flow.detach(), 'inverse_loss': inv.detach(), 'consistency_loss': cons.detach(), 'ltc_loss': ltc.detach(), 'static_ltc_loss': static_ltc.detach(), 'dynamic_ltc_loss': dynamic_ltc.detach(), 'dynamic_cost': dynamic_cost, 'dynamic_rollout_distance': dynamic_distance, 'sigreg_loss': reg.detach(), 'snapshot_round': path.new_tensor(snap), 'memory_size': path.new_tensor(0 if bank is None else bank.size(1)), **metrics}

def save_step_checkpoint(outdir, step, source, runtime, enc, cost, cfg):
    """Save independently loadable planner and LTC components mid-epoch."""
    stem = f"{cfg.output_model_name}_step_{step}"
    payload = checkpoint_payload(
        lewm_checkpoint=str(source['lewm_checkpoint']),
        action_block=runtime.action_block,
        flow=runtime.flow,
        inverse_dynamics=runtime.inverse_dynamics,
        cfg=OmegaConf.to_container(cfg, resolve=True),
    )
    payload['experience'] = {
        'phase': 'closed_loop_fine_tuning',
        'trajectory_encoder_state_dict': enc.state_dict(),
        'cost_model_state_dict': cost.state_dict(),
    }
    planner_path = outdir / f"{stem}.pt"
    torch.save(payload, planner_path)
    ltc_path = save_finetuned_ltc_components(
        run_dir=outdir,
        output_model_name=stem,
        epoch=step,
        lewm_checkpoint=str(source['lewm_checkpoint']),
        trajectory_encoder=enc,
        cost_model=cost,
        cfg=cfg,
    )
    print(
        f"step={step} checkpoint_done planner={planner_path} "
        f"ltc={ltc_path}",
        flush=True,
    )
@hydra.main(version_base=None, config_path='./config/train', config_name='fine_tuning')
def run(cfg: DictConfig):
    torch.manual_seed(int(cfg.seed))
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
    gen = torch.Generator(device=device).manual_seed(int(cfg.seed))
    (train, _) = make_loaders(cfg)
    runtime = LatentPlannerRuntime.from_checkpoint(cfg.planner_checkpoint, device=device)
    runtime.lewm = freeze(runtime.lewm)
    if int(runtime.action_block) != int(cfg.planner.action_block):
        raise ValueError('action_block differs from phase-one checkpoint')
    first = encode_latents(runtime.lewm, next(iter(train)), device)
    if first.size(1) != int(cfg.planner.horizon) + 1:
        raise ValueError('planner.horizon does not match data')
    (enc, cost) = load_ltc_components(cfg, first.size(-1), device)
    if enc.representation_dim != runtime.flow.path_feature_dim:
        raise ValueError('LTC representation_dim must equal flow.path_feature_dim')
    groups = [{'params': list(runtime.flow.parameters()) + list(runtime.inverse_dynamics.parameters()), 'lr': float(cfg.optimizer.planner_lr)}, {'params': enc.parameters(), 'lr': float(cfg.optimizer.encoder_lr)}, {'params': cost.parameters(), 'lr': float(cfg.optimizer.cost_lr)}]
    opt = torch.optim.AdamW(groups, weight_decay=float(cfg.optimizer.weight_decay))
    params = [p for g in groups for p in g['params']]
    sigreg = SIGReg(**cfg.loss.sigreg.kwargs).to(device) if float(cfg.loss.sigreg.weight) else None
    outdir = Path(stablewm_cache_dir(sub_folder='checkpoints'), cfg.subdir)
    outdir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, outdir / 'fine_tuning_config.yaml')
    wb = init_wandb(cfg, outdir)
    source = torch.load(cfg.planner_checkpoint, map_location='cpu', weights_only=False)
    step = 0
    try:
        for epoch in range(int(cfg.epochs)):
            runtime.flow.train()
            runtime.inverse_dynamics.train()
            enc.train()
            cost.train()
            for (i, batch) in enumerate(train):
                out = train_step(batch, runtime, enc, cost, sigreg, cfg, device, gen)
                opt.zero_grad(set_to_none=True)
                out['loss'].backward()
                norm = torch.nn.utils.clip_grad_norm_(params, float(cfg.grad_clip_norm)) if cfg.grad_clip_norm is not None else None
                opt.step()
                step += 1
                if step % int(cfg.log_interval) == 0:
                    print(f'epoch={epoch + 1} step={step} ' + ' '.join((f'{k}={metric_float(v):.4f}' for (k, v) in out.items())), flush=True)
                if wb is not None:
                    log = {f'train/{k}': metric_float(v) for (k, v) in out.items()} | {'train/epoch': epoch + 1}
                    if norm is not None:
                        log['train/grad_norm'] = float(norm)
                    wb.log(log, step=step)
                checkpoint_interval = cfg.get('checkpoint_interval_steps')
                if (
                    checkpoint_interval is not None
                    and step % int(checkpoint_interval) == 0
                ):
                    save_step_checkpoint(outdir, step, source, runtime, enc, cost, cfg)
                if (
                    cfg.max_train_batches is not None
                    and i + 1 >= int(cfg.max_train_batches)
                ):
                    break
            payload = checkpoint_payload(lewm_checkpoint=str(source['lewm_checkpoint']), action_block=runtime.action_block, flow=runtime.flow, inverse_dynamics=runtime.inverse_dynamics, cfg=OmegaConf.to_container(cfg, resolve=True))
            payload['experience'] = {'phase': 'closed_loop_fine_tuning', 'ltc_checkpoint': str(cfg.experience.ltc_checkpoint), 'trajectory_encoder_state_dict': enc.state_dict(), 'cost_model_state_dict': cost.state_dict()}
            latest = outdir / f'{cfg.output_model_name}.pt'
            torch.save(payload, outdir / f'{cfg.output_model_name}_epoch_{epoch + 1}.pt')
            torch.save(payload, latest)
            ltc_path = save_finetuned_ltc_components(run_dir=outdir, output_model_name=cfg.output_model_name, epoch=epoch + 1, lewm_checkpoint=str(source['lewm_checkpoint']), trajectory_encoder=enc, cost_model=cost, cfg=cfg)
            print(f'epoch={epoch + 1} checkpoint_done planner={latest} encoder={ep} cost={cp}', flush=True)
    finally:
        if wb is not None:
            wb.finish()


if __name__ == '__main__':
    run()
