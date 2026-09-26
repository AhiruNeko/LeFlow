"""Separate latent-trajectory encoder and learned trajectory cost predictor."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class AdaLN(nn.Module):
    """Goal-conditioned LayerNorm with zero-initialized modulation."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        shift, scale = self.modulation(condition).chunk(2, dim=-1)
        return self.norm(x) * (1 + scale[:, None]) + shift[:, None]


class GoalConditionedBlock(nn.Module):
    """Pre-AdaLN self-attention block for one latent trajectory."""

    def __init__(self, dim: int, heads: int, mlp_dim: int, dropout: float) -> None:
        super().__init__()
        self.attn_norm = AdaLN(dim)
        self.mlp_norm = AdaLN(dim)
        self.attn = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: Tensor, condition: Tensor, padding_mask: Tensor | None
    ) -> Tensor:
        attn_input = self.attn_norm(x, condition)
        update, _ = self.attn(
            attn_input,
            attn_input,
            attn_input,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        x = x + update
        return x + self.mlp(self.mlp_norm(x, condition))


class TrajectoryEncoder(nn.Module):
    """Encode a path and its true goal into a high-dimensional experience vector.

    The Transformer operates at ``model_dim``.  ``representation_dim`` is an
    independent output width, intentionally larger by default, for downstream
    cost prediction or experience retrieval.
    """

    def __init__(
        self,
        latent_dim: int,
        *,
        max_horizon: int = 20,
        model_dim: int = 256,
        representation_dim: int = 256,
        depth: int = 4,
        heads: int = 8,
        mlp_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if model_dim % heads:
            raise ValueError("model_dim must be divisible by heads")
        self.latent_dim = latent_dim
        self.max_horizon = max_horizon
        self.representation_dim = representation_dim

        # Tokens encode only their latent state. The task goal is supplied
        # separately through the goal-conditioned AdaLN blocks below.
        self.input_proj = nn.Linear(latent_dim, model_dim)
        self.goal_proj = nn.Sequential(
            nn.Linear(latent_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, model_dim))
        self.position = nn.Parameter(
            torch.randn(1, max_horizon + 2, model_dim) * 0.02
        )
        self.blocks = nn.ModuleList(
            GoalConditionedBlock(model_dim, heads, mlp_dim, dropout)
            for _ in range(depth)
        )
        self.output_norm = nn.LayerNorm(model_dim)
        self.representation_proj = nn.Sequential(
            nn.Linear(model_dim, representation_dim),
            nn.GELU(),
            nn.LayerNorm(representation_dim),
        )

    def forward(
        self,
        path: Tensor,
        goal: Tensor,
        *,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Return ``representation[B, representation_dim]``.

        ``path`` is ``[B, T, latent_dim]`` and may contain up to
        ``max_horizon + 1`` latent-state tokens.  ``goal`` is the true task
        goal, not the candidate path endpoint.
        """
        if path.ndim != 3 or goal.ndim != 2:
            raise ValueError("path must be [B, T, D] and goal must be [B, D]")
        batch, steps, dim = path.shape
        if dim != self.latent_dim or goal.shape != (batch, self.latent_dim):
            raise ValueError("path and goal latent dimensions must match")
        if steps > self.max_horizon + 1:
            raise ValueError("path exceeds max_horizon")
        if padding_mask is not None and padding_mask.shape != (batch, steps):
            raise ValueError("padding_mask must have shape [B, T]")

        x = self.input_proj(path)
        x = torch.cat((self.cls_token.expand(batch, -1, -1), x), dim=1)
        x = x + self.position[:, : steps + 1]

        if padding_mask is not None:
            cls_mask = torch.zeros(
                batch, 1, dtype=torch.bool, device=path.device
            )
            padding_mask = torch.cat((cls_mask, padding_mask), dim=1)

        condition = self.goal_proj(goal)
        for block in self.blocks:
            x = block(x, condition, padding_mask)
        return self.representation_proj(self.output_norm(x[:, 0]))


class TrajectoryCostModel(nn.Module):
    """Predict a lower-is-better scalar cost from an encoder representation.

    This class deliberately has no path encoder.  Use it as
    ``cost_model(trajectory_encoder(path, goal))``.
    """

    def __init__(self, representation_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        hidden_dim = max(representation_dim // 2, 1)
        self.representation_dim = representation_dim
        self.network = nn.Sequential(
            nn.Linear(representation_dim, representation_dim),
            nn.LayerNorm(representation_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(representation_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, representation: Tensor) -> Tensor:
        if representation.ndim != 2 or representation.size(-1) != self.representation_dim:
            raise ValueError(
                "representation must have shape [B, representation_dim]"
            )
        return self.network(representation).squeeze(-1)


@dataclass(frozen=True)
class TrajectoryCostLoss:
    total: Tensor
    regression: Tensor
    ranking: Tensor
    delta: Tensor
    endpoint: Tensor


def trajectory_cost_loss(
    predicted_cost: Tensor,
    target_cost: Tensor,
    *,
    representation: Tensor | None = None,
    path: Tensor | None = None,
    delta_head: nn.Module | None = None,
    endpoint_head: nn.Module | None = None,
    ranking_weight: float = 0.25,
    auxiliary_weight: float = 0.1,
    margin: float = 0.1,
) -> TrajectoryCostLoss:
    """Cost supervision with optional external encoder auxiliary heads.

    Keeping auxiliary heads outside ``TrajectoryCostModel`` preserves the
    encoder/cost-model separation.  They can be attached only while training
    the encoder to retain path information beyond scalar cost.
    """
    target_cost = target_cost.reshape_as(predicted_cost).to(predicted_cost)
    regression = F.smooth_l1_loss(predicted_cost, target_cost)

    true_lower = target_cost[:, None] < target_cost[None, :]
    predicted_difference = predicted_cost[:, None] - predicted_cost[None, :]
    ranking = (
        F.relu(margin + predicted_difference[true_lower]).mean()
        if true_lower.any()
        else predicted_cost.new_zeros(())
    )

    delta = predicted_cost.new_zeros(())
    endpoint = predicted_cost.new_zeros(())
    if any(item is not None for item in (representation, path, delta_head, endpoint_head)):
        if representation is None or path is None or delta_head is None or endpoint_head is None:
            raise ValueError("all auxiliary-loss inputs must be provided together")
        delta = F.mse_loss(delta_head(representation), path[:, -1] - path[:, 0])
        endpoint = F.mse_loss(endpoint_head(representation), path[:, -1])

    total = regression + ranking_weight * ranking + auxiliary_weight * (delta + endpoint)
    return TrajectoryCostLoss(total, regression, ranking, delta, endpoint)