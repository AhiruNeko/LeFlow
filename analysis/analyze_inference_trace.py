"""Trace generated paths, predicted costs, and frozen-LeWM rollout errors."""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import nn
import stable_pretraining as spt
import stable_worldmodel as swm
from utils import get_img_preprocessor
from latent_planner import LatentPlannerRuntime, load_ltc_components_for_evaluation
from latent_trajectory_cost import TrajectoryCostModel, TrajectoryEncoder
def parse_settings(value): return [tuple(map(int, item.strip().lower().split("x"))) for item in value.split(",") if item.strip()]


class LegacyDeltaTrajectoryEncoder(TrajectoryEncoder):
    """Checkpoint-local compatibility wrapper for old z + (goal-z) encoders."""

    def __init__(self, latent_dim, **architecture):
        super().__init__(latent_dim=latent_dim, **architecture)
        model_dim = self.cls_token.size(-1)
        self.input_proj = nn.Linear(2 * latent_dim, model_dim)

    def forward(self, path, goal, *, padding_mask=None):
        if path.ndim != 3 or goal.ndim != 2:
            raise ValueError("path must be [B, T, D] and goal must be [B, D]")
        batch, steps, dim = path.shape
        if dim != self.latent_dim or goal.shape != (batch, self.latent_dim):
            raise ValueError("path and goal latent dimensions must match")
        if steps > self.max_horizon + 1:
            raise ValueError("path exceeds max_horizon")
        if padding_mask is not None and padding_mask.shape != (batch, steps):
            raise ValueError("padding_mask must have shape [B, T]")
        x = self.input_proj(torch.cat((path, goal[:, None] - path), dim=-1))
        x = torch.cat((self.cls_token.expand(batch, -1, -1), x), dim=1)
        x = x + self.position[:, :steps + 1]
        if padding_mask is not None:
            cls_mask = torch.zeros(batch, 1, dtype=torch.bool, device=path.device)
            padding_mask = torch.cat((cls_mask, padding_mask), dim=1)
        condition = self.goal_proj(goal)
        for block in self.blocks:
            x = block(x, condition, padding_mask)
        return self.representation_proj(self.output_norm(x[:, 0]))


class LegacyUnboundedCostModel(nn.Module):
    """Compatibility model for old phase-2 scalar costs without sigmoid."""

    def __init__(self, representation_dim=256, dropout=.1):
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

    def forward(self, representation):
        return self.network(representation).squeeze(-1)


def load_standalone(path, component, latent_dim, device):
    """Load current or old separate phase-2 LTC components without repo changes."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("component") != component:
        raise ValueError(f"{path} is not a {component} checkpoint")
    architecture = dict(payload["architecture"])
    if component == "trajectory_encoder":
        input_dim = payload["state_dict"]["input_proj.weight"].size(1)
        cls = (
            LegacyDeltaTrajectoryEncoder
            if input_dim == 2 * latent_dim
            else TrajectoryEncoder
        )
        model = cls(latent_dim=latent_dim, **architecture)
    else:
        if architecture.get("output_activation") == "sigmoid":
            model = TrajectoryCostModel.from_checkpoint_architecture(architecture)
        else:
            model = LegacyUnboundedCostModel(
                representation_dim=int(architecture.get("representation_dim", 256)),
                dropout=float(architecture.get("dropout", .1)),
            )
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval().requires_grad_(False)

def load_ltc(args, latent_dim, device):
    if bool(args.encoder_checkpoint) != bool(args.cost_checkpoint):
        raise ValueError("provide both standalone LTC checkpoints or neither")
    if args.encoder_checkpoint:
        return (
            load_standalone(args.encoder_checkpoint, "trajectory_encoder", latent_dim, device),
            load_standalone(args.cost_checkpoint, "trajectory_cost", latent_dim, device),
        )
    return load_ltc_components_for_evaluation(
        args.planner_checkpoint, latent_dim=latent_dim, device=device
    )


@torch.no_grad()
def encode_score(paths, goal, encoder, cost):
    batch, count, length, latent = paths.shape
    flat = paths.reshape(batch * count, length, latent)
    goals = goal[:, None].expand(-1, count, -1).reshape(batch * count, latent)
    features = encoder(flat, goals).reshape(batch, count, -1)
    return features, cost(features.flatten(0, 1)).reshape(batch, count)


@torch.no_grad()
def sample_fixed_noise(runtime, z_start, z_goal, noise, flow_steps, path_features, path_costs):
    batch, latent = z_start.shape
    count = noise.size(0)
    if batch != 1 or noise.shape != (count, noise.size(1), latent):
        raise ValueError("fixed-noise analysis expects one task and [C, H-1, D] noise")
    z0 = z_start.expand(count, -1)
    zg = z_goal.expand(count, -1)
    x = noise.clone()
    dt = 1.0 / max(flow_steps, 1)
    for step in range(flow_steps):
        time = torch.full((count,), step * dt, device=x.device, dtype=x.dtype)
        expanded_features = path_features.expand(count, -1, -1) if path_features is not None else None
        expanded_costs = path_costs.expand(count, -1) if path_costs is not None else None
        x = x + runtime.flow(x, time, z0, zg, path_features=expanded_features, path_costs=expanded_costs) * dt
    return torch.cat((z0[:, None], x, zg[:, None]), dim=1)[None]

def make_trace_loader(args):
    """Dataset loader that preserves the expert action sequence for diagnostics."""
    dataset=swm.data.HDF5Dataset(
        name=args.dataset, frameskip=args.action_block, num_steps=args.horizon+1,
        keys_to_load=["pixels","action"], keys_to_cache=[], transform=None,
    )
    dataset.transform=spt.data.transforms.Compose(
        get_img_preprocessor(source="pixels",target="pixels",img_size=args.img_size)
    )
    return torch.utils.data.DataLoader(
        dataset,batch_size=args.batch_size,shuffle=True,num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),drop_last=True,
    )


@torch.no_grad()
def get_tasks(runtime,args,device):
    paths=[]; actions=[]; remaining=args.num_tasks
    for batch in make_trace_loader(args):
        current=runtime.encode_pixels(batch["pixels"].to(device))
        expert_action=batch["action"].to(device)
        paths.append(current[:remaining])
        actions.append(expert_action[:remaining,:args.horizon])
        remaining-=current.size(0)
        if remaining<=0: break
    if remaining>0: raise ValueError("not enough dataset paths")
    paths=torch.cat(paths)[:args.num_tasks]
    actions=torch.cat(actions)[:args.num_tasks]
    if paths.size(1)!=args.horizon+1: raise ValueError("dataset path length must equal horizon + 1")
    if actions.size(1)!=args.horizon: raise ValueError("expert action length must equal horizon")
    return paths,actions

@torch.no_grad()
def trace(task_id,expert,expert_actions,runtime,encoder,cost_model,candidates,rounds,noise,use_experience,args):
    start,goal=expert[None,0],expert[None,-1]
    paths=features=costs=None
    rows=[]; offset=0
    condition="experience_on" if use_experience else "experience_off"
    for round_id in range(1,rounds+1):
        generated=sample_fixed_noise(
            runtime,start,goal,noise[offset:offset+candidates],args.flow_steps,
            features,costs,
        )
        new_features,new_costs=encode_score(generated,goal,encoder,cost_model)
        actions=runtime.decode_actions(generated)
        final=runtime.rollout_final_latent(
            start,actions,history_size=args.lewm_history_size
        )
        errors=(final-goal[:,None]).square().mean(-1)
        action_errors=(actions[0]-expert_actions[None]).square().mean(dim=(1,2))
        memory_size=0 if paths is None else paths.size(1)
        for candidate in range(candidates):
            rows.append({
                "task_id":task_id,"setting":f"{candidates}x{rounds}",
                "condition":condition,"generated_path_index":offset+candidate+1,
                "initial_noise_index":offset+candidate+1,"round":round_id,
                "candidate_in_round":candidate+1,
                "memory_size_before_round":memory_size,
                "predicted_cost":float(new_costs[0,candidate]),
                "rollout_error":float(errors[0,candidate]),
                "expert_action_error":float(action_errors[candidate]),
                "is_min_cost":False,"is_min_rollout_error":False,
                "is_min_expert_action_error":False,
            })
        if use_experience:
            paths=generated if paths is None else torch.cat((paths,generated),1)
            features=new_features if features is None else torch.cat((features,new_features),1)
            costs=new_costs if costs is None else torch.cat((costs,new_costs),1)
            if paths.size(1)>args.experience_max_size:
                paths=paths[:,-args.experience_max_size:]
                features=features[:,-args.experience_max_size:]
                costs=costs[:,-args.experience_max_size:]
        offset+=candidates
    min(rows,key=lambda x:x["predicted_cost"])["is_min_cost"]=True
    min(rows,key=lambda x:x["rollout_error"])["is_min_rollout_error"]=True
    min(rows,key=lambda x:x["expert_action_error"])["is_min_expert_action_error"]=True
    return rows


def summary_for(rows):
    low_cost=next(x for x in rows if x["is_min_cost"])
    low_error=next(x for x in rows if x["is_min_rollout_error"])
    low_action=next(x for x in rows if x["is_min_expert_action_error"])
    return {
        "min_cost_index":low_cost["generated_path_index"],
        "min_cost":low_cost["predicted_cost"],
        "rollout_error_at_min_cost":low_cost["rollout_error"],
        "min_error_index":low_error["generated_path_index"],
        "min_rollout_error":low_error["rollout_error"],
        "cost_at_min_error":low_error["predicted_cost"],
        "min_expert_action_error_index":low_action["generated_path_index"],
        "min_expert_action_error":low_action["expert_action_error"],
        "rollout_error_at_min_expert_action_error":low_action["rollout_error"],
        "same_selected_path":low_cost["generated_path_index"]==low_error["generated_path_index"],
    }


def save_task_comparison(rows,path):
    """One fixed task/start-goal: overlay every CxR setting and both conditions."""
    try: import matplotlib.pyplot as plt
    except ImportError as error: raise RuntimeError("matplotlib is required") from error
    settings=list(dict.fromkeys(x["setting"] for x in rows))
    colors=plt.cm.tab10.colors
    fig,axes=plt.subplots(3,1,figsize=(13,10),sharex=True)
    for axis,key,ylabel,selected_key in (
        (axes[0],"predicted_cost","Predicted cost","is_min_cost"),
        (axes[1],"rollout_error","LeWM rollout error","is_min_rollout_error"),
        (axes[2],"expert_action_error","Expert action MSE","is_min_expert_action_error"),
    ):
        for index,setting in enumerate(settings):
            points=[x for x in rows if x["setting"]==setting]
            condition=points[0]["condition"]
            style="--" if condition=="experience_off" else "-"
            label="No experience baseline" if condition=="experience_off" else "Experience FIFO"
            color=colors[index % len(colors)]
            axis.plot(
                [x["initial_noise_index"] for x in points],
                [x[key] for x in points],color=color,linestyle=style,
                marker="o",markersize=2.5,linewidth=1.2,
                label=f"{setting} - {label}",
            )
            selected=next(x for x in points if x[selected_key])
            axis.scatter(
                selected["initial_noise_index"],selected[key],color=color,
                marker="*",edgecolors="#d62828",linewidths=.8,s=115,zorder=5,
            )
        axis.set_ylabel(ylabel); axis.grid(alpha=.25)
    axes[0].set_title(
        f"Task {rows[0]['task_id']}: fixed start/goal and matched initial latent noise"
    )
    axes[2].set_xlabel("Initial latent-noise index")
    axes[0].legend(loc="best",fontsize=8,ncol=2)
    fig.tight_layout()
    fig.savefig(path,dpi=190,bbox_inches="tight")
    plt.close(fig)

@torch.no_grad()
def run(args):
    settings=parse_settings(args.settings)
    if args.num_tasks<1: raise ValueError("num-tasks must be positive")
    if any(c>args.experience_max_size for c,_ in settings):
        raise ValueError("experience-max-size must fit one candidate round")
    torch.manual_seed(args.seed)
    device=torch.device(args.device if torch.cuda.is_available() else "cpu")
    runtime=LatentPlannerRuntime.from_checkpoint(args.planner_checkpoint,device=device)
    runtime.eval().requires_grad_(False)
    tasks,expert_actions=get_tasks(runtime,args,device)
    encoder,cost=load_ltc(args,tasks.size(-1),device)
    if args.horizon>encoder.max_horizon: raise ValueError("LTC horizon is too short")
    max_paths=max(c*r for c,r in settings)
    noise=torch.randn(
        tasks.size(0),max_paths,args.horizon-1,tasks.size(-1),device=device,
        dtype=tasks.dtype,generator=torch.Generator(device=device).manual_seed(args.seed+1),
    )
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    all_rows=[]; comparisons=[]
    for task_id,expert in enumerate(tasks):
        task_rows=[]
        for candidates,rounds in settings:
            count=candidates*rounds
            is_baseline=(candidates,rounds)==(64,1)
            rows=trace(task_id,expert,expert_actions[task_id],runtime,encoder,cost,candidates,rounds,
                       noise[task_id,:count],not is_baseline,args)
            stats=summary_for(rows)
            comparisons.append({
                "task_id":task_id,"setting":f"{candidates}x{rounds}",
                "condition":rows[0]["condition"],"num_paths":count,"summary":stats,
            })
            task_rows.extend(rows)
            print(f"task={task_id} setting={candidates}x{rounds} condition={rows[0]['condition']} min_error={stats['min_rollout_error']:.4f}",flush=True)
        task_csv=out/f"task_{task_id:03d}_all_settings_paths.csv"
        with task_csv.open("w",newline="") as file:
            writer=csv.DictWriter(file,fieldnames=list(task_rows[0])); writer.writeheader(); writer.writerows(task_rows)
        task_plot=out/f"task_{task_id:03d}_all_settings_comparison.png"
        save_task_comparison(task_rows,task_plot)
        for entry in comparisons:
            if entry["task_id"]==task_id:
                entry["task_csv"]=str(task_csv)
                entry["task_plot"]=str(task_plot)
        all_rows.extend(task_rows)
    all_csv=out/"all_paths.csv"
    with all_csv.open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)
    (out/"summary.json").write_text(json.dumps({
        "description":"All settings use fixed identical tasks and initial latent-noise indices. 64x1 is the no-experience baseline; every other setting uses FIFO experience. rollout_error follows frozen inverse-dynamics decoding and frozen LeWM rollout. expert_action_error is MSE to the aligned H-step expert action sequence from the dataset.",
        "checkpoints":{"planner":args.planner_checkpoint,"encoder":args.encoder_checkpoint or "embedded","cost":args.cost_checkpoint or "embedded"},
        "settings":{"dataset":args.dataset,"num_tasks":args.num_tasks,"horizon":args.horizon,"action_block":args.action_block,"candidate_round_settings":[f"{c}x{r}" for c,r in settings],"flow_steps":args.flow_steps,"experience_max_size":args.experience_max_size,"seed":args.seed},
        "comparisons":comparisons,"all_paths_csv":str(all_csv)},indent=2))
    print(f"summary_written={out/'summary.json'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner-checkpoint", required=True)
    parser.add_argument("--encoder-checkpoint")
    parser.add_argument("--cost-checkpoint")
    parser.add_argument("--dataset", default="pusht_expert_train")
    parser.add_argument("--num-tasks", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--action-block", type=int, default=5)
    parser.add_argument("--settings", default="64x1,16x4", help="Comma-separated CxR settings, e.g. 64x1,16x4")
    parser.add_argument("--experience-max-size", type=int, default=64)
    parser.add_argument("--flow-steps", type=int, default=16)
    parser.add_argument("--lewm-history-size", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())
