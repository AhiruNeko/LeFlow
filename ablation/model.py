"""Input ablations; shared Transformer blocks and trajectory head stay unchanged."""
from __future__ import annotations

from pathlib import Path

import torch

from reference_guided_world_model import ReferenceGuidedWorldModel

VARIANTS = ("no_local", "endpoints_actions")


class AblationWorldModel(ReferenceGuidedWorldModel):
    variant = ""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.variant not in VARIANTS:
            raise ValueError("Instantiate a concrete ablation model.")
        # Remove the entire local branch, including its learned biases. Keeping
        # these keys out of the state dict also prevents accidental full-model
        # loading by the original strict checkpoint reader.
        del self.local_prediction_projection
        del self.local_residual_projection
        del self.local_film

    def forward(self, reference_path, actions, local_predictions=None):
        # local_predictions is accepted for API compatibility, never consumed.
        if reference_path.ndim != 3 or actions.ndim != 3:
            raise ValueError("Expected [B,H+1,Dz] paths and [B,H,Da] actions.")
        batch, length, latent_dim = reference_path.shape
        horizon = length - 1
        if not 0 < horizon <= self.architecture.max_horizon:
            raise ValueError("Invalid horizon.")
        if latent_dim != self.architecture.latent_dim:
            raise ValueError("Invalid latent dimension.")
        if actions.shape != (batch, horizon, self.architecture.action_dim):
            raise ValueError("Actions must match the path horizon.")
        endpoints = torch.stack((reference_path[:, 0], reference_path[:, -1]), dim=1)
        condition = self.global_condition(endpoints.flatten(1))
        tokens = self.action_projection(actions) + self.action_position[:, :horizon]
        if self.variant == "no_local":
            memory = self.path_projection(reference_path) + self.path_position[:, :length]
        else:
            # Preserve cross-attention, but expose only the two task endpoints.
            # No interior planner state, path delta, or local residual is read.
            positions = torch.cat((self.path_position[:, :1], self.path_position[:, horizon:horizon + 1]), dim=1)
            memory = self.path_projection(endpoints) + positions
        for block in self.blocks:
            tokens = block(tokens, memory, condition, None)
        trajectory = endpoints[:, :1] + self.trajectory_decoder(tokens)
        return {
            "trajectory": trajectory,
            "global_condition": condition,
            "local_predictions": None,
            "local_residual": None,
        }

    @classmethod
    def from_checkpoint(cls, checkpoint, *, device="cpu"):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        variant = payload.get("config", {}).get("ablation", {}).get("variant")
        model_type = model_class(variant)
        if cls is not AblationWorldModel and cls.variant != variant:
            raise ValueError(f"Checkpoint ablation {variant!r} does not match {cls.variant!r}.")
        arch = payload["architecture"]
        model = model_type(
            latent_dim=int(arch["latent_dim"]), action_dim=int(arch["action_dim"]),
            max_horizon=int(arch["max_horizon"]), depth=int(arch["depth"]),
        )
        model.load_state_dict(payload["reference_guided_world_model_state_dict"], strict=True)
        return model.to(device).eval()


class NoLocalReferenceModel(AblationWorldModel):
    variant = "no_local"


class EndpointsActionsModel(AblationWorldModel):
    variant = "endpoints_actions"


def model_class(variant):
    models = {"no_local": NoLocalReferenceModel, "endpoints_actions": EndpointsActionsModel}
    if variant not in models:
        raise ValueError(f"Expected an ablation checkpoint/variant in {VARIANTS}; got {variant!r}.")
    return models[variant]


def training_step(model, frozen_lewm, reference_path, actions, executed_trajectory):
    """Keep equal-weight full-path MSE, without running local LeWM inference."""
    from reference_guided_world_model import continuous_trajectory_loss

    output = model(reference_path, actions)
    return {
        "loss": continuous_trajectory_loss(output["trajectory"], executed_trajectory),
        "trajectory": output["trajectory"],
        "local_predictions": None,
        "local_residual": None,
    }
