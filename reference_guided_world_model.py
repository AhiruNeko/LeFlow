"""Reference-guided local-to-global latent trajectory world model.

A frozen LeFlow planner supplies a reference path and one decoded action block per
segment. A frozen LeWM supplies independent local segment predictions. This
model predicts continuous states reached by execution of all action blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


def _compatible_heads(dim: int, preferred: int = 6) -> int:
    for heads in range(min(preferred, dim), 0, -1):
        if dim % heads == 0:
            return heads
    return 1


class _AdaLN(nn.Module):
    """Global task conditioning for one Transformer sublayer."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x, global_condition, local_film=None):
        gamma, beta = self.modulation(global_condition).chunk(2, dim=-1)
        while gamma.ndim < x.ndim:
            gamma, beta = gamma.unsqueeze(1), beta.unsqueeze(1)
        output = (1.0 + gamma) * self.norm(x) + beta
        if local_film is not None:
            local_gamma, local_beta = local_film
            output = (1.0 + local_gamma) * output + local_beta
        return output


class _StagedReferenceBlock(nn.Module):
    """Path cross-attention, temporal self-attention, then FFN."""

    def __init__(self, dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.query_cross_norm = _AdaLN(dim)
        self.reference_cross_norm = _AdaLN(dim)
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.self_norm = _AdaLN(dim)
        self.self_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ffn_norm = _AdaLN(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * dim, dim), nn.Dropout(dropout),
        )

    def forward(self, actions, reference, global_condition, local_film):
        query = self.query_cross_norm(actions, global_condition, local_film)
        memory = self.reference_cross_norm(reference, global_condition)
        cross, _ = self.cross_attention(query, memory, memory, need_weights=False)
        actions = actions + cross
        query = self.self_norm(actions, global_condition, local_film)
        self_out, _ = self.self_attention(query, query, query, need_weights=False)
        actions = actions + self_out
        return actions + self.ffn(self.ffn_norm(actions, global_condition, local_film))


@dataclass(frozen=True)
class ReferenceGuidedArchitecture:
    latent_dim: int
    action_dim: int
    max_horizon: int
    model_dim: int
    num_heads: int
    depth: int


class ReferenceGuidedWorldModel(nn.Module):
    """Action-primary local-to-global continuous latent world model.

    reference_path: [B, H+1, Dz], actions: [B, H, Da], and local_predictions:
    [B, H, Dz] where local_predictions[h] is frozen LeWM(p[h-1], a[h]).
    trajectory[:, h] predicts the continuous state after a[1] through a[h+1].
    """

    def __init__(self, latent_dim, action_dim, max_horizon, *, depth: int = 4, dropout: float = 0.0):
        super().__init__()
        if latent_dim <= 0 or action_dim <= 0 or max_horizon <= 0 or depth < 2:
            raise ValueError("invalid architecture dimensions")
        dim, heads = latent_dim, _compatible_heads(latent_dim)
        self.architecture = ReferenceGuidedArchitecture(latent_dim, action_dim, max_horizon, dim, heads, depth)

        self.action_projection = nn.Sequential(nn.LayerNorm(action_dim), nn.Linear(action_dim, dim))
        self.action_position = nn.Parameter(torch.zeros(1, max_horizon, dim))
        self.path_projection = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, dim))
        self.path_position = nn.Parameter(torch.zeros(1, max_horizon + 1, dim))

        self.global_condition = nn.Sequential(
            nn.LayerNorm(2 * latent_dim), nn.Linear(2 * latent_dim, dim),
            nn.SiLU(), nn.Linear(dim, dim),
        )
        self.local_prediction_projection = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, dim))
        self.local_residual_projection = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, dim))
        self.local_film = nn.Sequential(nn.SiLU(), nn.Linear(2 * dim, 2 * dim))
        self.blocks = nn.ModuleList([_StagedReferenceBlock(dim, heads, dropout) for _ in range(depth)])
        self.trajectory_decoder = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, latent_dim),
        )
        nn.init.normal_(self.action_position, std=0.02)
        nn.init.normal_(self.path_position, std=0.02)

    @classmethod
    def from_planner_runtime(cls, planner_runtime: Any, *, max_horizon: int | None = None):
        return cls(
            latent_dim=int(planner_runtime.flow.latent_dim),
            action_dim=int(planner_runtime.inverse_dynamics.action_dim),
            max_horizon=int(max_horizon or planner_runtime.flow.max_horizon),
        )

    @classmethod
    def from_planner_checkpoint(cls, planner_checkpoint: str, *, device: str | torch.device = "cpu"):
        from latent_planner import LatentPlannerRuntime
        runtime = LatentPlannerRuntime.from_checkpoint(planner_checkpoint, device=device)
        runtime.requires_grad_(False)
        runtime.eval()
        return cls.from_planner_runtime(runtime).to(device), runtime

    def forward(self, reference_path, actions, local_predictions):
        if reference_path.ndim != 3 or actions.ndim != 3 or local_predictions.ndim != 3:
            raise ValueError("all inputs must be rank-3 tensors")
        batch, length, latent_dim = reference_path.shape
        horizon = length - 1
        if not 0 < horizon <= self.architecture.max_horizon or latent_dim != self.architecture.latent_dim:
            raise ValueError("reference-path dimensions do not match the model")
        if actions.shape != (batch, horizon, self.architecture.action_dim):
            raise ValueError("action dimensions do not match reference-path segments")
        if local_predictions.shape != (batch, horizon, latent_dim):
            raise ValueError("local predictions do not match reference-path segments")

        global_condition = self.global_condition(torch.cat([reference_path[:, 0], reference_path[:, -1]], dim=-1))
        action_tokens = self.action_projection(actions) + self.action_position[:, :horizon]
        # Path memory intentionally contains absolute planner states only.
        # Action tokens encode transition intent; adding path deltas here would
        # duplicate that signal and blur the two roles.
        reference_tokens = (
            self.path_projection(reference_path)
            + self.path_position[:, :length]
        )

        local_residual = local_predictions - reference_path[:, 1:]
        local_features = torch.cat([
            self.local_prediction_projection(local_predictions),
            self.local_residual_projection(local_residual),
        ], dim=-1)
        local_film = self.local_film(local_features).chunk(2, dim=-1)

        for index, block in enumerate(self.blocks):
            action_tokens = block(
                action_tokens, reference_tokens, global_condition,
                None if index == 0 else local_film,
            )

        trajectory = reference_path[:, :1] + self.trajectory_decoder(action_tokens)
        return {
            "trajectory": trajectory,
            "local_predictions": local_predictions,
            "local_residual": local_residual,
            "global_condition": global_condition,
        }


@torch.no_grad()
def parallel_segment_predictions(lewm: nn.Module, reference_path: torch.Tensor, actions: torch.Tensor):
    """Evaluate every segment from its planner reference start independently."""
    batch, length, latent_dim = reference_path.shape
    horizon = length - 1
    if actions.shape[:2] != (batch, horizon):
        raise ValueError("actions must align with reference path")
    starts = reference_path[:, :-1].reshape(batch * horizon, 1, latent_dim)
    action_blocks = actions.reshape(batch * horizon, 1, actions.size(-1))
    embedding = lewm.action_encoder(action_blocks)
    return lewm.predict(starts, embedding)[:, -1].reshape(batch, horizon, latent_dim)


def continuous_trajectory_loss(predicted_trajectory: torch.Tensor, executed_trajectory: torch.Tensor):
    if predicted_trajectory.shape != executed_trajectory.shape:
        raise ValueError("prediction and target must match")
    return (predicted_trajectory - executed_trajectory).square().mean()


def segment_goal_score(predicted_trajectory, reference_path, *, segment_decay: float = 0.95):
    """Inference-only front-weighted candidate score."""
    if not 0.0 < segment_decay <= 1.0:
        raise ValueError("segment_decay must be in (0, 1]")
    if predicted_trajectory.shape != reference_path[:, 1:].shape:
        raise ValueError("trajectory and reference path do not align")
    horizon = predicted_trajectory.size(1)
    weights = torch.pow(
        torch.as_tensor(segment_decay, dtype=predicted_trajectory.dtype, device=predicted_trajectory.device),
        torch.arange(horizon, device=predicted_trajectory.device),
    )
    weights = weights / weights.sum().clamp_min(torch.finfo(weights.dtype).eps)
    return ((predicted_trajectory - reference_path[:, 1:]).square().mean(dim=-1) * weights).sum(dim=-1)



def reference_guided_training_step(
    model: ReferenceGuidedWorldModel,
    frozen_lewm: nn.Module,
    reference_path: torch.Tensor,
    actions: torch.Tensor,
    executed_trajectory: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """One supervised update from simulator-collected continuous trajectories.

    The executed trajectory must come from real environment execution followed
    by frozen-LeWM encoding. It must not be replaced by an autoregressive LeWM
    rollout, otherwise this model would only distil LeWM rather than learn its
    local-to-global error correction.
    """
    with torch.no_grad():
        local_predictions = parallel_segment_predictions(
            frozen_lewm, reference_path, actions
        )
    output = model(reference_path, actions, local_predictions)
    loss = continuous_trajectory_loss(output["trajectory"], executed_trajectory)
    return {
        "loss": loss,
        "trajectory": output["trajectory"],
        "local_predictions": local_predictions,
        "local_residual": output["local_residual"],
    }
