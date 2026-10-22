"""Original dataset evaluation with an ablation-only solver factory."""
import os
from pathlib import Path
from unittest.mock import patch

import hydra
from omegaconf import DictConfig

import eval as original_eval
import latent_planner
from ablation.solver import AblationLatentPathSolver

ROOT = Path(__file__).resolve().parents[1]
os.environ["LEFLOW_EVAL_CONFIG_DIR"] = str(ROOT / "config" / "eval")
os.environ["LEFLOW_ABLATION_DIR"] = str(ROOT / "ablation")


@hydra.main(version_base=None, config_path="config", config_name="eval_pusht")
def main(cfg: DictConfig):
    if cfg.solver._target_ != "latent_planner.LearnedLatentPathSolver":
        raise ValueError("Use solver=latent_flow; only its solver factory is substituted.")
    if cfg.eval.save_artifacts:
        raise ValueError("Keep eval.save_artifacts=false; the pipeline stores logs under ablation/.")
    with patch.object(latent_planner, "LearnedLatentPathSolver", AblationLatentPathSolver):
        original_eval.run.__wrapped__(cfg)


if __name__ == "__main__":
    main()
