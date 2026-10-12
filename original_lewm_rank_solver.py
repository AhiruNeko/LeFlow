from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from gymnasium.spaces import Box

from latent_planner import LatentPlannerRuntime, load_world_model


class OriginalLeWMRankedJointSolver:
    """Generate with a joint path--action flow and rank actions using original LeWM.

    The checkpoint's own frozen world model remains solely responsible for
    joint ODE generation.  A separate original autoregressive LeWM re-encodes
    the current frame and goal and ranks the resulting action candidates in
    its own latent space.  No latent vectors are transferred between models.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        original_lewm_checkpoint: str | Path,
        batch_size: int = 1,
        num_samples: int = 64,
        flow_steps: int = 16,
        history_size: int = 3,
        device: str | torch.device = "cpu",
        seed: int = 1234,
        world_model_checkpoint: str | Path | None = None,
        world_model_type: str | None = None,
        fast_source_dir: str | Path | None = None,
    ):
        self.checkpoint = checkpoint
        self.original_lewm_checkpoint = original_lewm_checkpoint
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.flow_steps = flow_steps
        self.history_size = history_size
        requested_device = torch.device(device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            requested_device = torch.device("cpu")
        self.device = requested_device

        # The checkpoint controls generation and may itself use Fast-LeWM.
        self.model = LatentPlannerRuntime.from_checkpoint(
            checkpoint,
            device=self.device,
            world_model_checkpoint=world_model_checkpoint,
            world_model_type=world_model_type,
            fast_source_dir=fast_source_dir,
        )
        if self.model.flow.action_dim is None:
            raise ValueError(
                "OriginalLeWMRankedJointSolver requires a joint path--action checkpoint."
            )

        self.original_lewm = load_world_model(
            original_lewm_checkpoint,
            world_model_type="lewm",
        ).to(self.device).eval()
        self.original_lewm.requires_grad_(False)
        self.torch_gen = torch.Generator(device=self.device).manual_seed(seed)

    def configure(self, *, action_space: gym.Space, n_envs: int, config: Any) -> None:
        if not isinstance(action_space, Box):
            raise TypeError(
                f"OriginalLeWMRankedJointSolver expects Box action space, got {type(action_space)}"
            )
        self._action_space = action_space
        self._n_envs = n_envs
        self._config = config
        self._action_dim = int(np.prod(action_space.shape[1:]))
        expected = self._action_dim * int(config.action_block)
        if self.model.flow.action_dim != expected:
            raise ValueError(
                "Joint checkpoint action-block dimension does not match the environment: "
                f"checkpoint={self.model.flow.action_dim}, expected={expected}."
            )

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def action_dim(self) -> int:
        return self._action_dim * self._config.action_block

    @property
    def horizon(self) -> int:
        return self._config.horizon

    @torch.no_grad()
    def _encode_original(self, pixels: torch.Tensor) -> torch.Tensor:
        if pixels.ndim == 4:
            pixels = pixels[:, None]
        return self.original_lewm.encode({"pixels": pixels.to(self.device)})["emb"][:, -1]

    @torch.no_grad()
    def _rank_with_original_lewm(
        self,
        info_dict: dict,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """Return B x S final-goal costs from autoregressive original LeWM."""
        batch_size, num_samples, horizon = actions.shape[:3]
        z_start = self._encode_original(info_dict["pixels"])
        z_goal = self._encode_original(info_dict["goal"])

        latent_history = (
            z_start[:, None, None, :]
            .expand(batch_size, num_samples, 1, -1)
            .reshape(batch_size * num_samples, 1, -1)
            .clone()
        )
        flat_actions = actions.reshape(batch_size * num_samples, horizon, -1).to(self.device)
        for step in range(horizon):
            action_history = flat_actions[
                :, max(0, step - self.history_size + 1) : step + 1
            ]
            state_history = latent_history[:, -action_history.size(1) :]
            action_embedding = self.original_lewm.action_encoder(action_history)
            next_latent = self.original_lewm.predict(state_history, action_embedding)[:, -1:]
            latent_history = torch.cat([latent_history, next_latent], dim=1)

        final_latent = latent_history[:, -1].reshape(batch_size, num_samples, -1)
        return F.mse_loss(
            final_latent,
            z_goal[:, None, :].expand_as(final_latent),
            reduction="none",
        ).mean(dim=-1)

    @torch.inference_mode()
    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        del init_action
        total_envs = len(next(iter(info_dict.values())))
        selected_actions: list[torch.Tensor] = []
        selected_costs: list[torch.Tensor] = []
        all_costs: list[torch.Tensor] = []

        for start in range(0, total_envs, self.batch_size):
            end = min(start + self.batch_size, total_envs)
            batch = {key: value[start:end] for key, value in info_dict.items()}
            z_start, z_goal = self.model.encode_current_and_goal(batch)
            _, actions = self.model.sample_trajectories(
                z_start,
                z_goal,
                horizon=self.horizon,
                num_samples=self.num_samples,
                flow_steps=self.flow_steps,
                generator=self.torch_gen,
            )
            costs = self._rank_with_original_lewm(batch, actions)
            best = costs.argmin(dim=1)
            batch_idx = torch.arange(actions.size(0), device=actions.device)
            selected_actions.append(actions[batch_idx, best].detach().cpu())
            selected_costs.append(costs[batch_idx, best].detach().cpu())
            all_costs.append(costs.detach().cpu())

        return {
            "actions": torch.cat(selected_actions, dim=0),
            "costs": torch.cat(selected_costs, dim=0).tolist(),
            "goal_costs": torch.cat(all_costs, dim=0),
            "rollout_count": total_envs * self.num_samples,
        }

    def __call__(self, *args: Any, **kwargs: Any) -> dict:
        return self.solve(*args, **kwargs)