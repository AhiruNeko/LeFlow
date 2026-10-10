from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import gymnasium as gym
import importlib.util
import sys

import numpy as np
import torch
from gymnasium.spaces import Box
from torch import nn
import torch.nn.functional as F

import stable_worldmodel as swm


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


def _load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_fast_lewm(
    checkpoint: str | Path,
    source_dir: str | Path | None = None,
) -> nn.Module:
    """Load a Fast-LeWM serialized object without polluting LeWM imports."""
    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"Fast-LeWM checkpoint not found: {checkpoint}")
    source_root = Path(source_dir) if source_dir else Path(__file__).parent / "third_party" / "fast_lewm_runtime"
    if not source_root.is_absolute() and not source_root.exists():
        source_root = Path(__file__).parent / source_root
    module_path = source_root / "module.py"
    jepa_path = source_root / "jepa.py"
    if not module_path.exists() or not jepa_path.exists():
        raise FileNotFoundError(
            "Fast-LeWM source is required to load its serialized checkpoint. "
            f"Expected module.py and jepa.py under {source_root}."
        )

    # The Fast checkpoint was pickled with top-level module names module
    # and jepa. Temporarily bind those names to Fast's implementation so
    # they cannot resolve to this repository's original-LeWM files.
    previous = {name: sys.modules.get(name) for name in ("module", "jepa")}
    try:
        _load_module_from_path("module", module_path)
        _load_module_from_path("jepa", jepa_path)
        model = torch.load(path, map_location="cpu", weights_only=False)
    finally:
        for name, old_module in previous.items():
            if old_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module
    if not isinstance(model, nn.Module):
        raise TypeError(f"Fast-LeWM checkpoint did not contain an nn.Module: {type(model)}")
    model.eval()
    model.requires_grad_(False)
    model.is_fast_lewm = True
    return model


def load_world_model(
    checkpoint: str | Path,
    *,
    world_model_type: str = "lewm",
    fast_source_dir: str | Path | None = None,
) -> nn.Module:
    kind = world_model_type.lower().replace("_", "-")
    if kind in {"lewm", "original", "original-lewm"}:
        model = load_lewm(checkpoint)
        model.is_fast_lewm = False
        return model
    if kind in {"fast", "fast-lewm", "fastlewm"}:
        return load_fast_lewm(checkpoint, source_dir=fast_source_dir)
    raise ValueError(
        f"Unknown world_model_type={world_model_type!r}; expected 'lewm' or 'fast-lewm'."
    )


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


class LatentPathFlow(nn.Module):
    """Joint rectified-flow model for latent-path interiors and action blocks.

    A path token and an action token share a transition index: the action token
    realizes the transition from the preceding latent state to that path token.
    The model therefore denoises a route and its action realization together,
    rather than decoding each transition afterwards with an independent MLP.

    action_dim=None retains the original path-only behavior. It is used
    only to load and evaluate pre-joint-flow checkpoints.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 512,
        depth: int = 4,
        max_horizon: int = 20,
        time_dim: int = 64,
        dropout: float = 0.0,
        action_dim: int | None = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.max_horizon = max_horizon
        self.depth = depth
        self.time_dim = time_dim
        self.dropout = dropout
        self.action_dim = action_dim
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_horizon - 1, hidden_dim) * 0.02
        )
        self.action_pos_embedding = (
            nn.Parameter(torch.randn(1, max_horizon, hidden_dim) * 0.02)
            if action_dim is not None
            else None
        )
        self.token_proj = nn.Linear(latent_dim, hidden_dim)
        self.action_token_proj = (
            nn.Linear(action_dim, hidden_dim) if action_dim is not None else None
        )
        self.path_type_embedding = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.action_type_embedding = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.start_proj = nn.Linear(latent_dim, hidden_dim)
        self.goal_proj = nn.Linear(latent_dim, hidden_dim)
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.net = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, latent_dim))
        self.action_out = (
            nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, action_dim))
            if action_dim is not None
            else None
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        z_start: torch.Tensor,
        z_goal: torch.Tensor,
        action_x_t: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        n_tokens = x_t.size(1)
        if n_tokens > self.max_horizon - 1:
            raise ValueError(
                f"Requested {n_tokens + 1} horizon, but max_horizon={self.max_horizon}"
            )
        cond = (
            self.start_proj(z_start)
            + self.goal_proj(z_goal)
            + self.time_embed(t.float())
        )
        path_tokens = self.token_proj(x_t)
        path_tokens = (
            path_tokens
            + self.pos_embedding[:, :n_tokens]
            + self.path_type_embedding
            + cond[:, None]
        )
        if action_x_t is None:
            return self.out(self.net(path_tokens))

        if self.action_dim is None or self.action_token_proj is None:
            raise ValueError(
                "action_x_t was supplied to a path-only LatentPathFlow. "
                "Construct it with action_dim to use the joint model."
            )
        if action_x_t.ndim != 3 or action_x_t.size(-1) != self.action_dim:
            raise ValueError(
                f"Expected action_x_t [B, H, {self.action_dim}], got "
                f"{tuple(action_x_t.shape)}."
            )
        if action_x_t.size(1) != n_tokens + 1:
            raise ValueError(
                "Joint flow requires one action block for every full path "
                f"transition, got {action_x_t.size(1)} blocks for {n_tokens + 1}."
            )

        action_tokens = self.action_token_proj(action_x_t)
        action_tokens = (
            action_tokens
            + self.action_pos_embedding[:, : action_x_t.size(1)]
            + self.action_type_embedding
            + cond[:, None]
        )
        # This is a route generator, not a causal world model: actions and
        # subgoals deliberately attend to the whole route.
        joint_tokens = self.net(torch.cat([path_tokens, action_tokens], dim=1))
        path_hidden = joint_tokens[:, :n_tokens]
        action_hidden = joint_tokens[:, n_tokens:]
        return self.out(path_hidden), self.action_out(action_hidden)


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
        inverse_dynamics: InverseDynamics | None,
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
        lewm = load_world_model(
            payload["lewm_checkpoint"],
            world_model_type=payload.get("world_model_type", "lewm"),
            fast_source_dir=payload.get("fast_source_dir"),
        )
        flow = LatentPathFlow(**arch["flow"])
        # Pre-joint-flow checkpoints lack the two zero-initialized token-type
        # embeddings introduced above. Their original path computation remains
        # unchanged, so allow only that legacy omission.
        flow.load_state_dict(
            payload["flow_state_dict"], strict=flow.action_dim is not None
        )
        inverse_dynamics = None
        inverse_arch = arch.get("inverse_dynamics")
        inverse_state = payload.get("inverse_dynamics_state_dict")
        if inverse_arch is not None and inverse_state is not None:
            inverse_dynamics = InverseDynamics(**inverse_arch)
            inverse_dynamics.load_state_dict(inverse_state)
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
    def sample_trajectories(
        self,
        z_start: torch.Tensor,
        z_goal: torch.Tensor,
        *,
        horizon: int,
        num_samples: int,
        flow_steps: int,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate aligned latent paths and action blocks from one ODE."""
        b, d = z_start.shape
        n = b * num_samples
        z0 = z_start[:, None].expand(b, num_samples, d).reshape(n, d)
        zg = z_goal[:, None].expand(b, num_samples, d).reshape(n, d)
        path_x = torch.randn(
            n,
            horizon - 1,
            d,
            device=z_start.device,
            dtype=z_start.dtype,
            generator=generator,
        )

        if self.flow.action_dim is None:
            dt = 1.0 / max(flow_steps, 1)
            for i in range(flow_steps):
                t = torch.full((n,), i * dt, device=z_start.device, dtype=z_start.dtype)
                path_x = path_x + self.flow(path_x, t, z0, zg) * dt
            paths = torch.cat([z0[:, None], path_x, zg[:, None]], dim=1)
            paths = paths.reshape(b, num_samples, horizon + 1, d)
            return paths, self.decode_actions(paths)

        action_x = torch.randn(
            n,
            horizon,
            self.flow.action_dim,
            device=z_start.device,
            dtype=z_start.dtype,
            generator=generator,
        )
        dt = 1.0 / max(flow_steps, 1)
        for i in range(flow_steps):
            t = torch.full((n,), i * dt, device=z_start.device, dtype=z_start.dtype)
            path_v, action_v = self.flow(path_x, t, z0, zg, action_x)
            path_x = path_x + path_v * dt
            action_x = action_x + action_v * dt
        paths = torch.cat([z0[:, None], path_x, zg[:, None]], dim=1)
        return (
            paths.reshape(b, num_samples, horizon + 1, d),
            action_x.reshape(b, num_samples, horizon, self.flow.action_dim),
        )

    @torch.no_grad()
    def sample_paths(
        self,
        z_start: torch.Tensor,
        z_goal: torch.Tensor,
        *,
        horizon: int,
        num_samples: int,
        flow_steps: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        paths, _ = self.sample_trajectories(
            z_start,
            z_goal,
            horizon=horizon,
            num_samples=num_samples,
            flow_steps=flow_steps,
            generator=generator,
        )
        return paths

    def decode_actions(self, paths: torch.Tensor) -> torch.Tensor:
        if self.inverse_dynamics is None:
            raise RuntimeError(
                "This joint-flow checkpoint has no standalone inverse dynamics model. "
                "Use sample_trajectories() to obtain its generated actions."
            )
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
        if getattr(self.lewm, "is_fast_lewm", False):
            act_emb = self.lewm.action_encoder(act, latent=z[:, :1])
            pred = self.lewm.predict(z[:, :1], act_emb)
            self.rollout_count += b * s
            return pred[:, -1].reshape(b, s, -1)
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
        paths, actions = self.sample_trajectories(
            z_start,
            z_goal,
            horizon=horizon,
            num_samples=num_samples,
            flow_steps=flow_steps,
            generator=generator,
        )
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
        if getattr(self.lewm, "is_fast_lewm", False):
            act_emb = self.lewm.action_encoder(act, latent=z[:, :1])
            pred = self.lewm.predict(z[:, :1], act_emb)
            return torch.cat([z, pred], dim=1).reshape(b, s, h + 1, -1)
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


def flow_matching_loss(
    flow: LatentPathFlow,
    z_path: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    z_start = z_path[:, 0]
    z_goal = z_path[:, -1]
    target = z_path[:, 1:-1]
    noise = torch.randn(target.shape, device=target.device, dtype=target.dtype, generator=generator)
    t = torch.rand(z_path.size(0), device=z_path.device, dtype=z_path.dtype, generator=generator)
    x_t = (1 - t[:, None, None]) * noise + t[:, None, None] * target
    pred_v = flow(x_t, t, z_start, z_goal)
    return F.mse_loss(pred_v, target - noise)


def joint_flow_matching_loss(
    flow: LatentPathFlow,
    z_path: torch.Tensor,
    actions: torch.Tensor,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flow-matching losses for aligned path interiors and action blocks.

    The same interpolation time is used for both streams.  pred_actions is
    the clean-action estimate induced by the predicted action velocity and is
    returned for the optional frozen-LeWM consistency loss.
    """
    if flow.action_dim is None:
        raise ValueError("joint_flow_matching_loss requires flow.action_dim.")
    if actions.size(1) != z_path.size(1) - 1:
        raise ValueError(
            "Expected one action block per path transition, got "
            f"{actions.size(1)} blocks for path length {z_path.size(1)}."
        )
    z_start = z_path[:, 0]
    z_goal = z_path[:, -1]
    path_target = z_path[:, 1:-1]
    path_noise = torch.randn(
        path_target.shape,
        device=path_target.device,
        dtype=path_target.dtype,
        generator=generator,
    )
    action_noise = torch.randn(
        actions.shape,
        device=actions.device,
        dtype=actions.dtype,
        generator=generator,
    )
    t = torch.rand(z_path.size(0), device=z_path.device, dtype=z_path.dtype, generator=generator)
    path_x_t = (1 - t[:, None, None]) * path_noise + t[:, None, None] * path_target
    action_x_t = (1 - t[:, None, None]) * action_noise + t[:, None, None] * actions
    path_v, action_v = flow(path_x_t, t, z_start, z_goal, action_x_t)
    loss_path = F.mse_loss(path_v, path_target - path_noise)
    loss_action = F.mse_loss(action_v, actions - action_noise)
    pred_actions = action_x_t + (1 - t[:, None, None]) * action_v
    return loss_path, loss_action, pred_actions


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
    """Compare generated action blocks with frozen world-model dynamics.

    Fast-LeWM predicts all action-prefix endpoints from z_0 in parallel;
    original LeWM retains its historical local autoregressive consistency loss.
    """
    if getattr(lewm, "is_fast_lewm", False):
        action_prefix_emb = lewm.action_encoder(
            pred_actions,
            latent=z_path[:, :1],
        )
        predicted_path = lewm.predict(z_path[:, :1], action_prefix_emb)
        return F.mse_loss(predicted_path, z_path[:, 1:])

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
    world_model_type: str = "lewm",
    fast_source_dir: str | None = None,
    action_block: int = 5,
    flow: LatentPathFlow,
    inverse_dynamics: InverseDynamics | None,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Serialize the joint model; old standalone-IDM payloads remain supported."""
    inverse_arch = None
    inverse_state = None
    if inverse_dynamics is not None:
        inverse_arch = {
            "latent_dim": inverse_dynamics.latent_dim,
            "action_dim": inverse_dynamics.action_dim,
            "hidden_dim": inverse_dynamics.hidden_dim,
            "depth": inverse_dynamics.depth,
            "dropout": inverse_dynamics.dropout,
        }
        inverse_state = inverse_dynamics.state_dict()
    return {
        "lewm_checkpoint": lewm_checkpoint,
        "world_model_type": world_model_type,
        "fast_source_dir": fast_source_dir,
        "action_block": action_block,
        "arch": {
            "flow": {
                "latent_dim": flow.latent_dim,
                "hidden_dim": flow.hidden_dim,
                "depth": flow.depth,
                "max_horizon": flow.max_horizon,
                "time_dim": flow.time_dim,
                "dropout": flow.dropout,
                "action_dim": flow.action_dim,
            },
            "inverse_dynamics": inverse_arch,
        },
        "config": cfg,
        "flow_state_dict": flow.state_dict(),
        "inverse_dynamics_state_dict": inverse_state,
    }
