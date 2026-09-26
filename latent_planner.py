from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
from torch import nn
import torch.nn.functional as F

import stable_worldmodel as swm

from latent_trajectory_cost import TrajectoryCostModel, TrajectoryEncoder


def stablewm_cache_dir(sub_folder: str | None = None) -> Path:
    try:
        return Path(swm.data.utils.get_cache_dir(sub_folder=sub_folder))
    except TypeError:
        root = Path(swm.data.utils.get_cache_dir())
        if sub_folder == "checkpoints":
            return root
        if sub_folder is None:
            return root
        return root / sub_folder


def _resolve_checkpoint(path_or_name: str | Path) -> Path:
    path = Path(path_or_name)
    if path.exists():
        return path

    base = stablewm_cache_dir(sub_folder="checkpoints")
    run_path = base / str(path_or_name)
    if run_path.exists() and run_path.is_file():
        return run_path
    pt_path = Path(f"{run_path}.pt")
    if pt_path.exists():
        return pt_path
    if run_path.is_dir():
        ckpts = sorted(
            list(run_path.glob("*.pt")) + list(run_path.glob("*_object.ckpt")),
            key=lambda x: x.stat().st_ctime,
            reverse=True,
        )
        if ckpts:
            return ckpts[0]

    object_path = Path(f"{run_path}_object.ckpt")
    if object_path.exists():
        return object_path

    raise FileNotFoundError(f"Could not resolve checkpoint: {path_or_name}")


def load_ltc_components_for_evaluation(
    ltc_checkpoint: str | Path,
    *,
    latent_dim: int,
    device: torch.device,
) -> tuple[TrajectoryEncoder, TrajectoryCostModel]:
    """Load frozen encoder and cost model from one combined LTC checkpoint."""
    payload = torch.load(ltc_checkpoint, map_location="cpu", weights_only=False)
    if payload.get("format") != "latent_trajectory_cost_v2":
        raise ValueError("ltc_checkpoint is not a combined LTC checkpoint")
    encoder_payload = payload["trajectory_encoder"]
    cost_payload = payload["cost_model"]
    encoder = TrajectoryEncoder(
        latent_dim=latent_dim, **dict(encoder_payload["architecture"])
    ).to(device)
    cost_model = TrajectoryCostModel(**dict(cost_payload["architecture"])).to(device)
    encoder.load_state_dict(encoder_payload["state_dict"], strict=True)
    cost_model.load_state_dict(cost_payload["state_dict"], strict=True)
    encoder.eval().requires_grad_(False)
    cost_model.eval().requires_grad_(False)
    return encoder, cost_model


def load_lewm(lewm_checkpoint: str | Path) -> nn.Module:
    path = Path(lewm_checkpoint)
    if not path.exists() and not path.is_absolute():
        cache_path = stablewm_cache_dir() / path
        if cache_path.exists():
            path = cache_path
    if path.exists() and path.is_file():
        module = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(module, (dict, OrderedDict)):
            state_dict = module.get("state_dict", module)
            return _build_lewm_from_state_dict(state_dict).eval()

        def scan(child):
            if hasattr(child, "get_cost"):
                return child.eval() if isinstance(child, nn.Module) else child
            if isinstance(child, nn.Module):
                for grandchild in child.children():
                    found = scan(grandchild)
                    if found is not None:
                        return found
            return None

        found = scan(module)
        if found is None:
            raise RuntimeError(f"No module with get_cost found in {lewm_checkpoint}")
        return found

    try:
        return swm.policy.AutoCostModel(str(lewm_checkpoint)).eval()
    except AttributeError:
        path = _resolve_checkpoint(lewm_checkpoint)
        module = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(module, (dict, OrderedDict)):
            state_dict = module.get("state_dict", module)
            return _build_lewm_from_state_dict(state_dict).eval()
        raise


def _build_lewm_from_state_dict(state_dict: dict[str, torch.Tensor]) -> nn.Module:
    import stable_pretraining as spt

    from jepa import JEPA
    from module import ARPredictor, Embedder, MLP

    state_dict = OrderedDict(
        (k.removeprefix("model."), v) for k, v in state_dict.items()
    )
    hidden_dim = state_dict["encoder.embeddings.cls_token"].shape[-1]
    patch_size = state_dict["encoder.embeddings.patch_embeddings.projection.weight"].shape[-1]
    num_patches = state_dict["encoder.embeddings.position_embeddings"].shape[1] - 1
    image_size = int(num_patches**0.5) * patch_size
    embed_dim = state_dict["projector.net.3.bias"].shape[0]
    projector_hidden_dim = state_dict["projector.net.0.bias"].shape[0]
    action_dim = state_dict["action_encoder.patch_embed.weight"].shape[1]
    smoothed_dim = state_dict["action_encoder.patch_embed.weight"].shape[0]
    action_mlp_hidden = state_dict["action_encoder.embed.0.bias"].shape[0]
    action_mlp_scale = max(action_mlp_hidden // embed_dim, 1)
    num_frames = state_dict["predictor.pos_embedding"].shape[1]
    predictor_depth = max(
        int(k.split(".")[3])
        for k in state_dict
        if k.startswith("predictor.transformer.layers.")
    ) + 1
    mlp_dim = state_dict["predictor.transformer.layers.0.mlp.net.1.bias"].shape[0]
    inner_dim = state_dict["predictor.transformer.layers.0.attn.to_out.0.weight"].shape[1]
    dim_head = 64 if inner_dim % 64 == 0 else hidden_dim
    heads = inner_dim // dim_head
    encoder_scale = {
        192: "tiny",
        384: "small",
        768: "base",
        1024: "large",
    }.get(hidden_dim)
    if encoder_scale is None:
        raise ValueError(f"Cannot infer ViT scale from hidden_dim={hidden_dim}")

    encoder = spt.backbone.utils.vit_hf(
        encoder_scale,
        patch_size=patch_size,
        image_size=image_size,
        pretrained=False,
        use_mask_token=False,
    )
    predictor = ARPredictor(
        num_frames=num_frames,
        depth=predictor_depth,
        heads=heads,
        mlp_dim=mlp_dim,
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        dim_head=dim_head,
        dropout=0.0,
        emb_dropout=0.0,
    )
    action_encoder = Embedder(
        input_dim=action_dim,
        smoothed_dim=smoothed_dim,
        emb_dim=embed_dim,
        mlp_scale=action_mlp_scale,
    )
    projector = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=projector_hidden_dim,
        norm_fn=torch.nn.BatchNorm1d,
    )
    pred_proj = MLP(
        input_dim=hidden_dim,
        output_dim=embed_dim,
        hidden_dim=projector_hidden_dim,
        norm_fn=torch.nn.BatchNorm1d,
    )
    model = JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        projector=projector,
        pred_proj=pred_proj,
    )
    model.load_state_dict(state_dict, strict=True)
    return model


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(10000.0, device=t.device))
            * torch.arange(half, device=t.device)
            / max(half - 1, 1)
        )
        args = t[:, None] * freqs[None]
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ExperienceConditionedFlowBlock(nn.Module):
    """Self-attention flow block augmented with experience cross-attention."""

    def __init__(self, hidden_dim: int, heads: int, dropout: float):
        super().__init__()
        self.self_norm = nn.LayerNorm(hidden_dim)
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        experience_keys: torch.Tensor | None,
        experience_values: torch.Tensor | None,
    ) -> torch.Tensor:
        self_input = self.self_norm(x)
        self_update, _ = self.self_attn(
            self_input, self_input, self_input, need_weights=False
        )
        x = x + self_update
        if experience_keys is not None:
            cross_query = self.cross_norm(x)
            cross_update, _ = self.cross_attn(
                cross_query,
                experience_keys,
                experience_values,
                need_weights=False,
            )
            x = x + cross_update
        return x + self.ffn(self.ffn_norm(x))


class LatentPathFlow(nn.Module):
    """Rectified-flow model with cost-conditioned experience cross-attention.

    ``path_features`` is an encoded sequence of prior candidate paths with
    shape ``[B, N, path_feature_dim]``. Its paired ``path_costs[B, N]`` is
    transformed into per-experience FiLM parameters that modulate *only* the
    cross-attention values; keys remain pure retrieval/similarity features.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 512,
        depth: int = 4,
        max_horizon: int = 20,
        time_dim: int = 64,
        dropout: float = 0.0,
        path_feature_dim: int = 256,
        heads: int = 8,
    ):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.max_horizon = max_horizon
        self.depth = depth
        self.time_dim = time_dim
        self.dropout = dropout
        self.path_feature_dim = path_feature_dim
        self.heads = heads
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_horizon - 1, hidden_dim) * 0.02
        )
        self.token_proj = nn.Linear(latent_dim, hidden_dim)
        self.start_proj = nn.Linear(latent_dim, hidden_dim)
        self.goal_proj = nn.Linear(latent_dim, hidden_dim)
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.path_feature_proj = nn.Linear(path_feature_dim, hidden_dim)
        self.cost_film = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        nn.init.zeros_(self.cost_film[-1].weight)
        nn.init.zeros_(self.cost_film[-1].bias)
        self.blocks = nn.ModuleList(
            ExperienceConditionedFlowBlock(hidden_dim, heads, dropout)
            for _ in range(depth)
        )
        self.out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, latent_dim))

    def _cost_conditioned_values(
        self, path_features: torch.Tensor, path_costs: torch.Tensor
    ) -> torch.Tensor:
        values = self.path_feature_proj(path_features)
        shift, scale = self.cost_film(path_costs[..., None].float()).chunk(2, dim=-1)
        return values * (1 + scale) + shift

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        z_start: torch.Tensor,
        z_goal: torch.Tensor,
        path_features: torch.Tensor | None = None,
        path_costs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n_tokens = x_t.size(1)
        if n_tokens > self.max_horizon - 1:
            raise ValueError(
                f"Requested {n_tokens + 1} horizon, but max_horizon={self.max_horizon}"
            )
        if (path_features is None) != (path_costs is None):
            raise ValueError("path_features and path_costs must be provided together")
        if path_features is not None:
            if path_features.ndim != 3 or path_features.shape[:2] != path_costs.shape:
                raise ValueError("path_features=[B,N,F] and path_costs=[B,N] are required")
            if path_features.size(0) != x_t.size(0):
                raise ValueError("path feature batch size must match x_t")
            if path_features.size(-1) != self.path_feature_dim:
                raise ValueError("path feature dimension must match path_feature_dim")
            experience_keys = self.path_feature_proj(path_features)
            experience_values = self._cost_conditioned_values(path_features, path_costs)
        else:
            experience_keys = None
            experience_values = None
        cond = self.start_proj(z_start) + self.goal_proj(z_goal) + self.time_embed(t.float())
        x = self.token_proj(x_t) + self.pos_embedding[:, :n_tokens] + cond[:, None]
        for block in self.blocks:
            x = block(x, experience_keys, experience_values)
        return self.out(x)


class InverseDynamics(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        depth: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.dropout = dropout
        layers: list[nn.Module] = []
        in_dim = latent_dim * 3
        for i in range(depth):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z_t: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_t, z_next, z_next - z_t], dim=-1)
        return self.net(x)


class LatentPlannerRuntime(nn.Module):
    def __init__(
        self,
        lewm: nn.Module,
        flow: LatentPathFlow,
        inverse_dynamics: InverseDynamics,
        action_block: int = 5,
    ):
        super().__init__()
        self.lewm = lewm.eval()
        self.flow = flow
        self.inverse_dynamics = inverse_dynamics
        self.action_block = action_block
        self.rollout_count = 0
        self.lewm.requires_grad_(False)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @classmethod
    def from_checkpoint(cls, checkpoint: str | Path, device: str | torch.device = "cpu"):
        path = _resolve_checkpoint(checkpoint)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise TypeError(
                "Latent planner checkpoints must be dicts with state_dict entries, "
                "not serialized model objects."
            )

        arch = payload["arch"]
        lewm = load_lewm(payload["lewm_checkpoint"])
        flow = LatentPathFlow(**arch["flow"])
        inverse_dynamics = InverseDynamics(**arch["inverse_dynamics"])
        flow.load_state_dict(payload["flow_state_dict"])
        inverse_dynamics.load_state_dict(payload["inverse_dynamics_state_dict"])
        model = cls(
            lewm=lewm,
            flow=flow,
            inverse_dynamics=inverse_dynamics,
            action_block=payload.get("action_block", 1),
        )
        return model.to(device).eval()

    @torch.no_grad()
    def encode_pixels(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.ndim == 4:
            pixels = pixels[:, None]
        info = {"pixels": pixels.to(self.device)}
        return self.lewm.encode(info)["emb"]

    @torch.no_grad()
    def encode_current_and_goal(self, info_dict: dict) -> tuple[torch.Tensor, torch.Tensor]:
        pixels = info_dict["pixels"].to(self.device)
        goal = info_dict["goal"].to(self.device)
        z_start = self.encode_pixels(pixels)[:, -1]
        z_goal = self.encode_pixels(goal)[:, -1]
        return z_start, z_goal

    @torch.no_grad()
    def sample_paths(
        self,
        z_start: torch.Tensor,
        z_goal: torch.Tensor,
        *,
        horizon: int,
        num_samples: int,
        flow_steps: int,
        path_features: torch.Tensor | None = None,
        path_costs: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if (path_features is None) != (path_costs is None):
            raise ValueError("path_features and path_costs must be provided together")
        if path_features is not None and path_features.size(0) != z_start.size(0):
            raise ValueError("experience batch size must match z_start batch size")
        b, d = z_start.shape
        n = b * num_samples
        z0 = z_start[:, None].expand(b, num_samples, d).reshape(n, d)
        zg = z_goal[:, None].expand(b, num_samples, d).reshape(n, d)
        x = torch.randn(
            n,
            horizon - 1,
            d,
            device=z_start.device,
            dtype=z_start.dtype,
            generator=generator,
        )
        dt = 1.0 / max(flow_steps, 1)
        for i in range(flow_steps):
            t = torch.full((n,), i * dt, device=z_start.device, dtype=z_start.dtype)
            expanded_features = (
                path_features.repeat_interleave(num_samples, dim=0)
                if path_features is not None
                else None
            )
            expanded_costs = (
                path_costs.repeat_interleave(num_samples, dim=0)
                if path_costs is not None
                else None
            )
            x = x + self.flow(
                x,
                t,
                z0,
                zg,
                path_features=expanded_features,
                path_costs=expanded_costs,
            ) * dt
        start = z0[:, None]
        goal = zg[:, None]
        return torch.cat([start, x, goal], dim=1).reshape(b, num_samples, horizon + 1, d)

    @torch.no_grad()
    def plan_experience_guided(
        self,
        info_dict: dict,
        *,
        trajectory_encoder: TrajectoryEncoder,
        cost_model: TrajectoryCostModel,
        horizon: int,
        samples_per_round: int,
        rounds: int,
        experience_max_size: int,
        min_experience_size: int,
        top_k: int,
        cost_threshold: float | None,
        flow_steps: int,
        history_size: int = 3,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Generate candidate paths over multiple FIFO-memory flow rounds.

        LTC costs choose a small candidate set only.  The final action plan is
        selected independently by the original LeFlow inverse-dynamics + LeWM
        rollout-to-goal score, so an LTC calibration error cannot directly
        become the executed action.
        """
        if samples_per_round < 1 or rounds < 1:
            raise ValueError("samples_per_round and rounds must both be positive")
        if experience_max_size < 0 or top_k < 1:
            raise ValueError("experience_max_size must be >= 0 and top_k must be positive")
        if not 0 <= min_experience_size <= experience_max_size:
            raise ValueError(
                "min_experience_size must satisfy 0 <= min_experience_size <= experience_max_size"
            )
        if trajectory_encoder.representation_dim != self.flow.path_feature_dim:
            raise ValueError(
                "LTC representation_dim must equal the flow path_feature_dim"
            )

        z_start, z_goal = self.encode_current_and_goal(info_dict)
        batch_size = z_start.size(0)
        memory_features: torch.Tensor | None = None
        memory_costs: torch.Tensor | None = None
        top_paths: torch.Tensor | None = None
        top_costs: torch.Tensor | None = None
        rounds_used = 0
        bootstrap_samples = 0

        def append_to_memory(
            features: torch.Tensor, costs: torch.Tensor
        ) -> None:
            nonlocal memory_features, memory_costs
            if not experience_max_size:
                return
            appended_features = (
                features
                if memory_features is None
                else torch.cat((memory_features, features), dim=1)
            )
            appended_costs = (
                costs
                if memory_costs is None
                else torch.cat((memory_costs, costs), dim=1)
            )
            memory_features = appended_features[:, -experience_max_size:]
            memory_costs = appended_costs[:, -experience_max_size:]

        for round_index in range(rounds):
            current_memory_size = (
                0 if memory_features is None else memory_features.size(1)
            )
            missing = max(min_experience_size - current_memory_size, 0)
            if missing:
                bootstrap_paths = self.sample_paths(
                    z_start,
                    z_goal,
                    horizon=horizon,
                    num_samples=missing,
                    flow_steps=flow_steps,
                    generator=generator,
                )
                flat_bootstrap = bootstrap_paths.flatten(0, 1)
                bootstrap_goals = z_goal[:, None].expand(
                    batch_size, missing, -1
                ).reshape(-1, z_goal.size(-1))
                bootstrap_features = trajectory_encoder(
                    flat_bootstrap, bootstrap_goals
                ).reshape(batch_size, missing, -1)
                bootstrap_costs = cost_model(
                    bootstrap_features.flatten(0, 1)
                ).reshape(batch_size, missing)
                append_to_memory(bootstrap_features, bootstrap_costs)
                bootstrap_samples += missing

            candidates = self.sample_paths(
                z_start,
                z_goal,
                horizon=horizon,
                num_samples=samples_per_round,
                flow_steps=flow_steps,
                path_features=memory_features,
                path_costs=memory_costs,
                generator=generator,
            )
            flat_paths = candidates.flatten(0, 1)
            flat_goals = z_goal[:, None].expand(
                batch_size, samples_per_round, -1
            ).reshape(-1, z_goal.size(-1))
            candidate_features = trajectory_encoder(flat_paths, flat_goals).reshape(
                batch_size, samples_per_round, -1
            )
            candidate_costs = cost_model(candidate_features.flatten(0, 1)).reshape(
                batch_size, samples_per_round
            )

            # FIFO memory contains only paths generated for this same
            # start/goal condition. The newest candidates replace the oldest.
            append_to_memory(candidate_features, candidate_costs)

            candidate_pool = candidates if top_paths is None else torch.cat(
                [top_paths, candidates], dim=1
            )
            cost_pool = candidate_costs if top_costs is None else torch.cat(
                [top_costs, candidate_costs], dim=1
            )
            keep = min(top_k, cost_pool.size(1))
            top_costs, top_index = torch.topk(
                cost_pool, k=keep, dim=1, largest=False, sorted=True
            )
            gather_index = top_index[:, :, None, None].expand(
                -1, -1, candidate_pool.size(2), candidate_pool.size(3)
            )
            top_paths = candidate_pool.gather(1, gather_index)
            rounds_used = round_index + 1

            # A batch must be collectively good enough before stopping.  This
            # retains the vectorized policy interface without starving harder
            # environments of their remaining sampling rounds.
            # Do not allow threshold-based termination before a full top-k
            # candidate set exists. In particular, when fewer than top_k
            # paths are sampled in the first round, its mean is not yet the
            # requested top-k mean.
            if (
                cost_threshold is not None
                and top_costs.size(1) == top_k
                and bool((top_costs.mean(dim=1) <= cost_threshold).all())
            ):
                break

        assert top_paths is not None and top_costs is not None
        top_actions = self.decode_actions(top_paths)
        rollout_final = self.rollout_final_latent(
            z_start, top_actions, history_size=history_size
        )
        rollout_goal_costs = F.mse_loss(
            rollout_final,
            z_goal[:, None].expand_as(rollout_final),
            reduction="none",
        ).mean(dim=-1)
        best = rollout_goal_costs.argmin(dim=1)
        batch_index = torch.arange(batch_size, device=self.device)
        return {
            "actions": top_actions[batch_index, best].detach().cpu(),
            "costs": rollout_goal_costs[batch_index, best].detach().cpu(),
            "goal_costs": rollout_goal_costs.detach().cpu(),
            "experience_top_costs": top_costs.detach().cpu(),
            "rounds_used": torch.full(
                (batch_size,), rounds_used, device=self.device, dtype=torch.long
            ).cpu(),
            "bootstrap_samples": torch.full(
                (batch_size,), bootstrap_samples, device=self.device, dtype=torch.long
            ).cpu(),
        }

    def decode_actions(self, paths: torch.Tensor) -> torch.Tensor:
        z_t = paths[..., :-1, :]
        z_next = paths[..., 1:, :]
        flat_actions = self.inverse_dynamics(
            z_t.reshape(-1, z_t.size(-1)),
            z_next.reshape(-1, z_next.size(-1)),
        )
        return flat_actions.reshape(*z_t.shape[:-1], -1)

    @torch.no_grad()
    def rollout_final_latent(
        self,
        z_start: torch.Tensor,
        actions: torch.Tensor,
        history_size: int = 3,
    ) -> torch.Tensor:
        b, s, h = actions.shape[:3]
        z = z_start[:, None, None].expand(b, s, 1, -1).reshape(b * s, 1, -1).clone()
        act = actions.reshape(b * s, h, -1).to(self.device)
        for t in range(h):
            act_hist = act[:, max(0, t - history_size + 1) : t + 1]
            emb_hist = z[:, -act_hist.size(1) :]
            act_emb = self.lewm.action_encoder(act_hist)
            pred = self.lewm.predict(emb_hist, act_emb)[:, -1:]
            z = torch.cat([z, pred], dim=1)
        self.rollout_count += b * s
        return z[:, -1].reshape(b, s, -1)

    @torch.no_grad()
    def plan(
        self,
        info_dict: dict,
        *,
        horizon: int,
        num_samples: int,
        flow_steps: int,
        score_mode: str = "rollout_goal",
        goal_weight: float = 1.0,
        consistency_weight: float = 0.0,
        smoothness_weight: float = 0.0,
        history_size: int = 3,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        z_start, z_goal = self.encode_current_and_goal(info_dict)
        paths = self.sample_paths(
            z_start,
            z_goal,
            horizon=horizon,
            num_samples=num_samples,
            flow_steps=flow_steps,
            generator=generator,
        )
        actions = self.decode_actions(paths)
        score_mode = score_mode.lower()
        if score_mode == "rollout_goal":
            rollout_final = self.rollout_final_latent(z_start, actions, history_size)
            goal_cost = F.mse_loss(
                rollout_final,
                z_goal[:, None].expand_as(rollout_final),
                reduction="none",
            ).mean(dim=-1)
            cost = goal_weight * goal_cost
        elif score_mode == "first":
            # No reranking ablation: execute the first sampled candidate as-is.
            goal_cost = torch.zeros(actions.shape[:2], device=actions.device, dtype=actions.dtype)
            cost = goal_cost.clone()
        elif score_mode == "path_smoothness":
            goal_cost = torch.zeros(actions.shape[:2], device=actions.device, dtype=actions.dtype)
            accel = paths[:, :, 2:] - 2 * paths[:, :, 1:-1] + paths[:, :, :-2]
            cost = accel.pow(2).mean(dim=(-1, -2))
        else:
            raise ValueError(
                f"Unknown latent planner score_mode={score_mode!r}. "
                "Expected one of: rollout_goal, first, path_smoothness."
            )

        if consistency_weight:
            rollout_paths = self.rollout_paths(z_start, actions, history_size)
            consistency = (rollout_paths - paths).pow(2).mean(dim=(-1, -2))
            cost = cost + consistency_weight * consistency

        if smoothness_weight:
            accel = paths[:, :, 2:] - 2 * paths[:, :, 1:-1] + paths[:, :, :-2]
            smooth = accel.pow(2).mean(dim=(-1, -2))
            cost = cost + smoothness_weight * smooth

        best = cost.argmin(dim=1)
        batch_idx = torch.arange(actions.size(0), device=actions.device)
        return {
            "actions": actions[batch_idx, best].detach().cpu(),
            "costs": cost[batch_idx, best].detach().cpu(),
            "all_costs": cost.detach().cpu(),
            "goal_costs": goal_cost.detach().cpu(),
        }

    @torch.no_grad()
    def rollout_paths(
        self,
        z_start: torch.Tensor,
        actions: torch.Tensor,
        history_size: int = 3,
    ) -> torch.Tensor:
        b, s, h = actions.shape[:3]
        z = z_start[:, None, None].expand(b, s, 1, -1).reshape(b * s, 1, -1).clone()
        act = actions.reshape(b * s, h, -1).to(self.device)
        for t in range(h):
            act_hist = act[:, max(0, t - history_size + 1) : t + 1]
            emb_hist = z[:, -act_hist.size(1) :]
            act_emb = self.lewm.action_encoder(act_hist)
            pred = self.lewm.predict(emb_hist, act_emb)[:, -1:]
            z = torch.cat([z, pred], dim=1)
        return z.reshape(b, s, h + 1, -1)


class LearnedLatentPathSolver:
    """stable-worldmodel solver wrapper for the learned latent planner."""

    def __init__(
        self,
        checkpoint: str | Path,
        batch_size: int = 1,
        num_samples: int = 64,
        flow_steps: int = 16,
        score_mode: str = "rollout_goal",
        goal_weight: float = 1.0,
        consistency_weight: float = 0.0,
        smoothness_weight: float = 0.0,
        device: str | torch.device = "cpu",
        seed: int = 1234,
        history_size: int = 3,
    ):
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.flow_steps = flow_steps
        self.score_mode = score_mode
        self.goal_weight = goal_weight
        self.consistency_weight = consistency_weight
        self.smoothness_weight = smoothness_weight
        requested_device = torch.device(device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            requested_device = torch.device("cpu")
        self.device = requested_device
        self.history_size = history_size
        self.model = LatentPlannerRuntime.from_checkpoint(checkpoint, device=self.device)
        self.torch_gen = torch.Generator(device=self.device).manual_seed(seed)

    def configure(self, *, action_space: gym.Space, n_envs: int, config: Any) -> None:
        self._action_space = action_space
        self._n_envs = n_envs
        self._config = config
        self._action_dim = int(np.prod(action_space.shape[1:]))
        if not isinstance(action_space, Box):
            raise TypeError(f"LearnedLatentPathSolver expects Box action space, got {type(action_space)}")

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def action_dim(self) -> int:
        return self._action_dim * self._config.action_block

    @property
    def horizon(self) -> int:
        return self._config.horizon

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        return self.solve(*args, **kwargs)

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        del init_action
        total_envs = len(next(iter(info_dict.values())))
        all_actions = []
        all_costs = []
        all_goal_costs = []
        for start in range(0, total_envs, self.batch_size):
            end = min(start + self.batch_size, total_envs)
            batch = {k: v[start:end] for k, v in info_dict.items()}
            out = self.model.plan(
                batch,
                horizon=self.horizon,
                num_samples=self.num_samples,
                flow_steps=self.flow_steps,
                score_mode=self.score_mode,
                goal_weight=self.goal_weight,
                consistency_weight=self.consistency_weight,
                smoothness_weight=self.smoothness_weight,
                history_size=self.history_size,
                generator=self.torch_gen,
            )
            all_actions.append(out["actions"])
            all_costs.append(out["costs"])
            all_goal_costs.append(out["goal_costs"])
        return {
            "actions": torch.cat(all_actions, dim=0),
            "costs": torch.cat(all_costs, dim=0).tolist(),
            "goal_costs": torch.cat(all_goal_costs, dim=0),
            "rollout_count": self.model.rollout_count,
        }


class ExperienceGuidedLatentPathSolver(LearnedLatentPathSolver):
    """FIFO experience-guided multi-round flow sampler for evaluation."""

    def __init__(
        self,
        *,
        rounds: int = 4,
        samples_per_round: int = 16,
        experience_max_size: int = 64,
        min_experience_size: int = 0,
        top_k: int = 8,
        cost_threshold: float | None = 1.0,
        ltc_checkpoint: str | Path | None = None,
        **kwargs: Any,
    ):
        super().__init__(num_samples=samples_per_round, **kwargs)
        if rounds < 1:
            raise ValueError("rounds must be positive")
        payload = torch.load(
            _resolve_checkpoint(self.checkpoint), map_location="cpu", weights_only=False
        )
        experience = payload.get("experience", {})
        ltc_path = ltc_checkpoint or experience.get("ltc_checkpoint")
        if not ltc_path:
            raise ValueError(
                "Experience-guided evaluation requires an LTC checkpoint, either "
                "in planner metadata or solver config."
            )
        self.trajectory_encoder, self.cost_model = load_ltc_components_for_evaluation(
            ltc_path,
            latent_dim=self.model.flow.latent_dim,
            device=self.device,
        )
        if self.trajectory_encoder.representation_dim != self.model.flow.path_feature_dim:
            raise ValueError(
                "LTC representation_dim does not match the planner checkpoint "
                "flow.path_feature_dim."
            )
        self.rounds = rounds
        self.samples_per_round = samples_per_round
        self.experience_max_size = experience_max_size
        self.min_experience_size = min_experience_size
        self.top_k = top_k
        self.cost_threshold = cost_threshold

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        del init_action
        total_envs = len(next(iter(info_dict.values())))
        all_actions, all_costs, all_goal_costs = [], [], []
        all_experience_costs, all_rounds, all_bootstrap_samples = [], [], []
        for start in range(0, total_envs, self.batch_size):
            end = min(start + self.batch_size, total_envs)
            batch = {key: value[start:end] for key, value in info_dict.items()}
            out = self.model.plan_experience_guided(
                batch,
                trajectory_encoder=self.trajectory_encoder,
                cost_model=self.cost_model,
                horizon=self.horizon,
                samples_per_round=self.samples_per_round,
                rounds=self.rounds,
                experience_max_size=self.experience_max_size,
                min_experience_size=self.min_experience_size,
                top_k=self.top_k,
                cost_threshold=self.cost_threshold,
                flow_steps=self.flow_steps,
                history_size=self.history_size,
                generator=self.torch_gen,
            )
            all_actions.append(out["actions"])
            all_costs.append(out["costs"])
            all_goal_costs.append(out["goal_costs"])
            all_experience_costs.append(out["experience_top_costs"])
            all_rounds.append(out["rounds_used"])
            all_bootstrap_samples.append(out["bootstrap_samples"])
        return {
            "actions": torch.cat(all_actions, dim=0),
            "costs": torch.cat(all_costs, dim=0).tolist(),
            "goal_costs": torch.cat(all_goal_costs, dim=0),
            "experience_top_costs": torch.cat(all_experience_costs, dim=0),
            "experience_rounds_used": torch.cat(all_rounds, dim=0),
            "experience_bootstrap_samples": torch.cat(all_bootstrap_samples, dim=0),
            "rollout_count": self.model.rollout_count,
        }


def flow_matching_loss(
    flow: LatentPathFlow,
    z_path: torch.Tensor,
    path_features: torch.Tensor | None = None,
    path_costs: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    z_start = z_path[:, 0]
    z_goal = z_path[:, -1]
    target = z_path[:, 1:-1]
    noise = torch.randn(target.shape, device=target.device, dtype=target.dtype, generator=generator)
    t = torch.rand(z_path.size(0), device=z_path.device, dtype=z_path.dtype, generator=generator)
    x_t = (1 - t[:, None, None]) * noise + t[:, None, None] * target
    pred_v = flow(
        x_t,
        t,
        z_start,
        z_goal,
        path_features=path_features,
        path_costs=path_costs,
    )
    return F.mse_loss(pred_v, target - noise)


def inverse_dynamics_loss(
    inverse_dynamics: InverseDynamics,
    z_path: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred = inverse_dynamics(z_path[:, :-1].reshape(-1, z_path.size(-1)), z_path[:, 1:].reshape(-1, z_path.size(-1)))
    pred = pred.reshape(z_path.size(0), z_path.size(1) - 1, -1)
    return F.mse_loss(pred, actions), pred


def lewm_consistency_loss(
    lewm: nn.Module,
    z_path: torch.Tensor,
    pred_actions: torch.Tensor,
    history_size: int = 3,
) -> torch.Tensor:
    losses = []
    for t in range(pred_actions.size(1)):
        start = max(0, t - history_size + 1)
        emb_hist = z_path[:, start : t + 1]
        act_hist = pred_actions[:, start : t + 1]
        act_emb = lewm.action_encoder(act_hist)
        pred = lewm.predict(emb_hist, act_emb)[:, -1]
        losses.append(F.mse_loss(pred, z_path[:, t + 1]))
    return torch.stack(losses).mean()


def smoothness_loss(z_path: torch.Tensor) -> torch.Tensor:
    if z_path.size(1) < 3:
        return z_path.new_tensor(0.0)
    accel = z_path[:, 2:] - 2 * z_path[:, 1:-1] + z_path[:, :-2]
    return accel.pow(2).mean()


def checkpoint_payload(
    *,
    lewm_checkpoint: str,
    action_block: int,
    flow: LatentPathFlow,
    inverse_dynamics: InverseDynamics,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    return {
        "lewm_checkpoint": lewm_checkpoint,
        "action_block": action_block,
        "arch": {
            "flow": {
                "latent_dim": flow.latent_dim,
                "hidden_dim": flow.hidden_dim,
                "depth": flow.depth,
                "max_horizon": flow.max_horizon,
                "time_dim": flow.time_dim,
                "dropout": flow.dropout,
                "path_feature_dim": flow.path_feature_dim,
                "heads": flow.heads,
            },
            "inverse_dynamics": {
                "latent_dim": inverse_dynamics.latent_dim,
                "action_dim": inverse_dynamics.action_dim,
                "hidden_dim": inverse_dynamics.hidden_dim,
                "depth": inverse_dynamics.depth,
                "dropout": inverse_dynamics.dropout,
            },
        },
        "config": cfg,
        "flow_state_dict": flow.state_dict(),
        "inverse_dynamics_state_dict": inverse_dynamics.state_dict(),
    }
