"""Reuse the staged trainer with process-local ablation dependencies only."""
from contextlib import contextmanager
import os
from pathlib import Path
from unittest.mock import patch

import hydra
from omegaconf import DictConfig

import train_reference_guided_world_model as trainer
from latent_planner import LatentPlannerRuntime
from ablation.model import model_class, training_step

ROOT = Path(__file__).resolve().parents[1]
os.environ["LEFLOW_TRAIN_CONFIG_DIR"] = str(ROOT / "config" / "train")
os.environ["LEFLOW_ABLATION_DIR"] = str(ROOT / "ablation")


@contextmanager
def training_dependencies(cfg):
    selected = model_class(str(cfg.ablation.variant))

    class ConfiguredModel(selected):
        @classmethod
        def from_planner_checkpoint(cls, checkpoint, *, device="cpu"):
            runtime = LatentPlannerRuntime.from_checkpoint(
                checkpoint, device=device, lewm_checkpoint=cfg.lewm_checkpoint
            ).requires_grad_(False).eval()
            return cls.from_planner_runtime(runtime).to(device), runtime

    # Restore both bindings on exit, including exceptions. No source file or
    # default model is modified; optimizer, data split, labels and selection
    # criteria are all executed by the original trainer.
    with patch.object(trainer, "ReferenceGuidedWorldModel", ConfiguredModel), patch.object(
        trainer, "reference_guided_training_step", training_step
    ):
        yield


def run_training(cfg):
    if cfg.staged_training.generate_rollout_cache:
        raise ValueError("Ablations read an existing rollout cache; generation must stay disabled.")
    output = Path(str(cfg.subdir)).expanduser().resolve()
    if not output.is_relative_to(ROOT / "ablation"):
        raise ValueError("Ablation outputs must stay inside the repository ablation/ directory.")
    with training_dependencies(cfg):
        return trainer.run.__wrapped__(cfg)


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig):
    run_training(cfg)


if __name__ == "__main__":
    main()
