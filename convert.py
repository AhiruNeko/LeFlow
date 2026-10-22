import argparse
import json
from pathlib import Path

import torch
import stable_pretraining as spt

from jepa import JEPA
from module import ARPredictor, Embedder, MLP


def build_model(cfg: dict) -> JEPA:
    encoder = spt.backbone.utils.vit_hf(
        cfg["encoder"]["size"],
        patch_size=cfg["encoder"]["patch_size"],
        image_size=cfg["encoder"]["image_size"],
        pretrained=False,
        use_mask_token=False,
    )

    def mlp(key: str) -> MLP:
        return MLP(
            input_dim=cfg[key]["input_dim"],
            output_dim=cfg[key]["output_dim"],
            hidden_dim=cfg[key]["hidden_dim"],
            norm_fn=torch.nn.BatchNorm1d,
        )

    predictor_cfg = cfg["predictor"].copy()
    predictor_cfg.pop("_target_", None)
    action_cfg = cfg["action_encoder"].copy()
    action_cfg.pop("_target_", None)
    return JEPA(
        encoder=encoder,
        predictor=ARPredictor(**predictor_cfg),
        action_encoder=Embedder(**action_cfg),
        projector=mlp("projector"),
        pred_proj=mlp("pred_proj"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a LeWM state dict into an AutoCostModel checkpoint."
    )
    parser.add_argument("--weights", type=Path, required=True, help="LeWM weights.pt state dict.")
    parser.add_argument("--config", type=Path, required=True, help="Matching LeWM config.json.")
    parser.add_argument("--output", type=Path, required=True, help="Destination model checkpoint.")
    args = parser.parse_args()

    cfg = json.loads(args.config.read_text())
    model = build_model(cfg)
    state_dict = torch.load(args.weights, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.eval(), args.output)
    print(f"Converted {args.weights} -> {args.output}")


if __name__ == "__main__":
    main()
