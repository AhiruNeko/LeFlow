import json, torch, stable_pretraining as spt
from pathlib import Path
from jepa import JEPA
from module import ARPredictor, Embedder, MLP
import stable_worldmodel as swm

import os
os.environ['STABLEWM_HOME'] = '/root/projects/pusht'

src = Path(swm.data.utils.get_cache_dir(), "checkpoints/pusht")
out = Path(swm.data.utils.get_cache_dir(), "checkpoints/pusht", "latent_planner.ckpt")

cfg = json.loads((src / "latent_planner_config.yaml").read_text())
encoder = spt.backbone.utils.vit_hf(
    cfg["encoder"]["size"],
    patch_size=cfg["encoder"]["patch_size"],
    image_size=cfg["encoder"]["image_size"],
    pretrained=False, use_mask_token=False,
)
mlp = lambda k: MLP(input_dim=cfg[k]["input_dim"], output_dim=cfg[k]["output_dim"],
                    hidden_dim=cfg[k]["hidden_dim"], norm_fn=torch.nn.BatchNorm1d)

predictor_cfg = cfg["predictor"].copy()
predictor_cfg.pop("_target_", None)

action_cfg = cfg["action_encoder"].copy()
action_cfg.pop("_target_", None)

model = JEPA(
    encoder=encoder,
    predictor=ARPredictor(**predictor_cfg),
    action_encoder=Embedder(**action_cfg),
    projector=mlp("projector"),
    pred_proj=mlp("pred_proj"),
)
sd = torch.load(src / "latent_planner.pt", map_location="cpu", weights_only=False)
model.load_state_dict(sd, strict=True)
out.parent.mkdir(parents=True, exist_ok=True)
torch.save(model, out)
