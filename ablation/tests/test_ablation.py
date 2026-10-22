"""Behavioral checks for removed inputs and unchanged staged orchestration."""
import copy
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from ablation.model import AblationWorldModel, EndpointsActionsModel, NoLocalReferenceModel, training_step
from ablation.solver import AblationPlannerRuntime, AblationLatentPathSolver
from ablation.train import run_training
from ablation.analyze import measure_task, save_reports
from reference_guided_world_model import ReferenceGuidedWorldModel
import train_reference_guided_world_model as trainer

ROOT = Path(__file__).resolve().parents[2]
ABLATION = ROOT / "ablation"


class ToyLeWM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.action_encoder = torch.nn.Linear(2, 12, bias=False)

    def predict(self, latent, actions):
        return latent + 0.1 * actions


class ToyRuntime(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lewm = ToyLeWM()
        self.flow = SimpleNamespace(latent_dim=12, max_horizon=4)
        self.inverse_dynamics = torch.nn.Linear(12, 2)
        self.inverse_dynamics.action_dim = 2
        self.action_block = 1

    def encode_pixels(self, pixels):
        return pixels

    def sample_paths(self, start, goal, *, horizon, num_samples, flow_steps):
        alpha = torch.linspace(0, 1, horizon + 1, device=start.device)
        path = start[:, None] + alpha[None, :, None] * (goal - start)[:, None]
        return path[:, None]

    def decode_actions(self, paths):
        return paths[:, :, 1:, :2] - paths[:, :, :-1, :2]


class AblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(7)
        self.path = torch.randn(2, 5, 12)
        self.actions = torch.randn(2, 4, 2)
        self.local = torch.randn(2, 4, 12)

    def test_local_values_cannot_affect_either_model(self):
        for kind in (NoLocalReferenceModel, EndpointsActionsModel):
            model = kind(12, 2, 4).eval()
            a = model(self.path, self.actions, self.local)["trajectory"]
            b = model(self.path, self.actions, torch.full_like(self.local, float("nan")))["trajectory"]
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertFalse(any("local_" in key for key in model.state_dict()))

    def test_only_second_variant_ignores_interior_path(self):
        changed = self.path.clone()
        changed[:, 1:-1] = torch.randn_like(changed[:, 1:-1]) * 100
        for kind in (NoLocalReferenceModel, EndpointsActionsModel):
            model = kind(12, 2, 4).eval()
            a = model(self.path, self.actions)["trajectory"]
            b = model(changed, self.actions)["trajectory"]
            self.assertEqual(torch.equal(a, b), kind is EndpointsActionsModel)
        path = self.path.clone().requires_grad_(True)
        model = EndpointsActionsModel(12, 2, 4)
        model(path, self.actions)["trajectory"].square().mean().backward()
        self.assertEqual(float(path.grad[:, 1:-1].abs().max()), 0.0)
        self.assertGreater(float(path.grad[:, 0].abs().sum()), 0.0)
        self.assertGreater(float(path.grad[:, -1].abs().sum()), 0.0)

    def test_actions_and_endpoints_still_affect_predictions(self):
        model = EndpointsActionsModel(12, 2, 4).eval()
        original = model(self.path, self.actions)["trajectory"]
        self.assertFalse(torch.equal(original, model(self.path, -self.actions)["trajectory"]))
        changed = self.path.clone()
        changed[:, -1] = torch.randn_like(changed[:, -1])
        self.assertFalse(torch.equal(original, model(changed, self.actions)["trajectory"]))

    def test_shared_backbone_is_unchanged(self):
        torch.manual_seed(10)
        full = ReferenceGuidedWorldModel(12, 2, 4)
        torch.manual_seed(10)
        ablated = NoLocalReferenceModel(12, 2, 4)
        for name, weight in ablated.state_dict().items():
            torch.testing.assert_close(weight, full.state_dict()[name], rtol=0, atol=0)

    def test_step_has_gradients_and_does_not_call_lewm(self):
        model = NoLocalReferenceModel(12, 2, 4)
        output = training_step(model, object(), self.path, self.actions, torch.randn(2, 4, 12))
        output["loss"].backward()
        self.assertGreater(float(model.trajectory_decoder[-1].weight.grad.abs().sum()), 0.0)

    def test_runtime_skips_local_lewm(self):
        runtime = AblationPlannerRuntime(torch.nn.Identity(), torch.nn.Linear(12, 12), torch.nn.Linear(12, 2))
        model = EndpointsActionsModel(12, 2, 4).eval()
        paths = self.path[:, None].expand(-1, 3, -1, -1)
        actions = self.actions[:, None].expand(-1, 3, -1, -1)
        out = runtime.reference_guided_rollout_paths(model, paths, actions)
        self.assertEqual(tuple(out.shape), (2, 3, 4, 12))
        torch.testing.assert_close(out[:, 0], model(self.path, self.actions)["trajectory"])

    def test_checkpoint_identifies_variant_and_rejects_full_loader(self):
        for kind in (NoLocalReferenceModel, EndpointsActionsModel):
            model = kind(12, 2, 4).eval()
            with tempfile.TemporaryDirectory(dir=ABLATION) as tmp:
                path = Path(tmp) / "model.pt"
                cfg = OmegaConf.create({"ablation": {"variant": kind.variant}})
                trainer.checkpoint(model, cfg, path, {})
                loaded = AblationWorldModel.from_checkpoint(path)
                self.assertIsInstance(loaded, kind)
                torch.testing.assert_close(loaded(self.path, self.actions)["trajectory"], model(self.path, self.actions)["trajectory"])
                with self.assertRaises(RuntimeError):
                    ReferenceGuidedWorldModel.from_checkpoint(path)
                wrong = EndpointsActionsModel if kind is NoLocalReferenceModel else NoLocalReferenceModel
                with self.assertRaises(ValueError):
                    wrong.from_checkpoint(path)

    def test_original_two_stage_runner_for_both_variants(self):
        original_factory = trainer.ReferenceGuidedWorldModel
        original_step = trainer.reference_guided_training_step
        for variant in ("no_local", "endpoints_actions"):
            with tempfile.TemporaryDirectory(dir=ABLATION) as tmp:
                root = Path(tmp)
                planner = root / "planner.pt"
                torch.save({"config": {"planner": {"horizon": 2}}}, planner)
                cache = root / "cache.h5"
                with h5py.File(cache, "w") as stream:
                    stream.attrs.update(complete=True, horizon=2, latent_dim=12,
                                        action_dim=2, model_max_horizon=4, source_records=4)
                    stream["reference_path"] = np.random.default_rng(1).normal(size=(4, 3, 12)).astype("float16")
                    stream["actions"] = np.random.default_rng(2).normal(size=(4, 2, 2)).astype("float16")
                    stream["executed_target"] = np.random.default_rng(3).normal(size=(4, 2, 12)).astype("float16")
                with initialize_config_dir(version_base=None, config_dir=str(ABLATION / "config")):
                    cfg = compose(config_name="train", overrides=[
                        f"ablation.variant={variant}", f"ablation.output_dir={root / 'run'}",
                        f"planner_checkpoint={planner}", "lewm_checkpoint=explicit.pt",
                        f"rollout_cache.path={cache}", "device=cpu", "loader.num_workers=0",
                        "loader.batch_size=2", "wandb.enabled=false", "max_train_batches=1",
                        "validation.max_batches=1", "staged_training.distillation_epochs=1",
                        "staged_training.fine_tune_epochs=1",
                    ])
                source = [{"pixels": torch.randn(3, 12)} for _ in range(40)]
                runtime = ToyRuntime()
                with patch("ablation.train.LatentPlannerRuntime.from_checkpoint", return_value=runtime) as load, patch.object(
                    trainer, "build_datasets", return_value=(None, source)
                ), patch.object(trainer, "create_rollout_backend", side_effect=AssertionError("Cache generation is forbidden")):
                    run_training(cfg)
                    self.assertEqual(load.call_args.kwargs["lewm_checkpoint"], "explicit.pt")
                prefix = root / "run" / "train" / f"ablation_{variant}"
                for suffix in ("_distilled.pt", "_distillation_best.pt", "_best.pt", ".pt"):
                    self.assertTrue(Path(str(prefix) + suffix).is_file())
                loaded = AblationWorldModel.from_checkpoint(str(prefix) + "_best.pt")
                self.assertEqual(loaded.variant, variant)
                self.assertIs(trainer.ReferenceGuidedWorldModel, original_factory)
                self.assertIs(trainer.reference_guided_training_step, original_step)

    def test_three_way_reports_use_same_ground_truth(self):
        target = torch.randn(4, 3, 12)
        data = {"target": target, "goal": torch.randn(4, 12),
                "lewm": target + 0.5, "reference_guided": target + 0.2,
                "ablation": target.clone()}
        records, task = measure_task(data, 0, 5, 10)
        self.assertEqual(len(records), 4)
        self.assertEqual(records[0]["ablation_full_mse"], 0.0)
        self.assertAlmostEqual(records[0]["lewm_full_mse"], 0.25, places=6)
        self.assertAlmostEqual(task["ablation_pearson"], 1.0)
        self.assertEqual(task["ablation_selection_regret"], 0.0)
        with tempfile.TemporaryDirectory(dir=ABLATION) as tmp:
            save_reports(Path(tmp), SimpleNamespace(test=True), records, [task])
            for filename in ("candidate_metrics.csv", "task_ranking_metrics.csv", "summary.json", "mse_comparison.png", "goal_score_correlation.png"):
                self.assertTrue((Path(tmp) / filename).is_file())

    @unittest.skipUnless((Path('/root/projects/pusht/leflow/pusht/latent_planner_h10.pt')).is_file(), "Local planner unavailable")
    def test_real_planner_checkpoint_solver_loads_and_scores(self):
        planner = "/root/projects/pusht/leflow/pusht/latent_planner_h10.pt"
        lewm = "/root/projects/pusht/checkpoints/pusht/lewm_object.ckpt"
        with tempfile.TemporaryDirectory(dir=ABLATION) as tmp:
            checkpoint = Path(tmp) / "ablation.pt"
            model = EndpointsActionsModel(192, 10, 20).eval()
            trainer.checkpoint(model, OmegaConf.create({"ablation": {"variant": "endpoints_actions"}}), checkpoint, {})
            solver = AblationLatentPathSolver(
                checkpoint=planner, lewm_checkpoint=lewm, reference_guided_checkpoint=checkpoint,
                device="cpu", num_samples=2, flow_steps=1,
            )
            path = torch.randn(1, 2, 11, 192)
            actions = torch.randn(1, 2, 10, 10)
            with patch.object(solver.model.lewm, "predict", side_effect=AssertionError("No local teacher call")):
                output = solver.model.reference_guided_rollout_paths(solver.reference_guided_model, path, actions)
            self.assertEqual(tuple(output.shape), (1, 2, 10, 192))


if __name__ == "__main__":
    unittest.main()
