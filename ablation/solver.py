"""Ablation-only solver; keep planner, IDM, ranking, and policy execution intact."""
import torch

from latent_planner import LatentPlannerRuntime, LearnedLatentPathSolver, _resolve_checkpoint
from ablation.model import AblationWorldModel
from reference_guided_world_model import continuous_lewm_predictions


class AblationPlannerRuntime(LatentPlannerRuntime):
    @torch.no_grad()
    def reference_guided_rollout_paths(self, reference_guided_model, paths, actions):
        batch, samples, length, dim = paths.shape
        horizon = length - 1
        if actions.shape[:3] != (batch, samples, horizon):
            raise ValueError("Candidate actions and paths do not align.")
        flat_paths = paths.reshape(batch * samples, length, dim)
        flat_actions = actions.reshape(batch * samples, horizon, actions.size(-1))
        condition = None
        if reference_guided_model.variant == "continuous_condition":
            condition = continuous_lewm_predictions(self.lewm, flat_paths, flat_actions)
        output = reference_guided_model(flat_paths, flat_actions, condition)["trajectory"]
        return output.reshape(batch, samples, horizon, dim)


class AblationLatentPathSolver(LearnedLatentPathSolver):
    def __init__(self, *args, reference_guided_checkpoint=None, **kwargs):
        if not reference_guided_checkpoint:
            raise ValueError("An ablation checkpoint is required.")
        # The parent still constructs exactly the same frozen planner/IDM/LeWM.
        super().__init__(*args, reference_guided_checkpoint=None, **kwargs)
        runtime = self.model
        self.model = AblationPlannerRuntime(
            runtime.lewm, runtime.flow, runtime.inverse_dynamics, runtime.action_block
        ).to(self.device).eval()
        self.reference_guided_model = AblationWorldModel.from_checkpoint(
            str(_resolve_checkpoint(reference_guided_checkpoint)), device=self.device
        ).requires_grad_(False)
        arch = self.reference_guided_model.architecture
        if arch.latent_dim != self.model.flow.latent_dim:
            raise ValueError("Ablation latent dimension differs from the planner.")
        if arch.action_dim != self.model.inverse_dynamics.action_dim:
            raise ValueError("Ablation action dimension differs from the IDM.")
        if arch.max_horizon < self.model.flow.max_horizon:
            raise ValueError("Ablation maximum horizon is smaller than the planner capacity.")
