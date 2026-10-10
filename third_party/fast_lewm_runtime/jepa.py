"""JEPA Implementation"""

import ast

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v

class JEPA(nn.Module):

    def __init__(
        self,
        encoder,
        predictor,
        action_encoder,
        projector=None,
        pred_proj=None,
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        self.consistency_loss_weight = 0.0
        self.action_num_blocks_per_step = None

    @staticmethod
    def _initial_action_condition_latent(action, latent):
        return latent[:, :1]

    def encode(self, info):
        """Encode observations and actions into embeddings.
        info: dict with pixels and action keys
        """

        pixels = info['pixels'].float()
        b = pixels.size(0)
        pixels = rearrange(pixels, "b t ... -> (b t) ...") # flatten for encoding
        output = self.encoder(pixels, interpolate_pos_encoding=True)
        pixels_emb = output.last_hidden_state[:, 0]  # cls token
        emb = self.projector(pixels_emb)
        info["emb"] = rearrange(emb, "(b t) d -> b t d", b=b)

        if "action" in info:
            # Condition the whole action sequence only on the initial latent z0.
            action_cond_latent = self._initial_action_condition_latent(
                info["action"], info["emb"]
            )
            info["act_emb"] = self.action_encoder(
                info["action"], latent=action_cond_latent
            )

        return info

    def predict(self, emb, act_emb):
        """Predict next state embedding
        emb:
            - (B, D)
            - (B, T, D)
        act_emb:
            - (B, F, A_emb)
            - (B, T, A_emb) (inference compatibility)
        """
        preds = self.predictor(emb, act_emb)
        if preds.dim() == 2:
            return self.pred_proj(preds)

        if preds.dim() == 3:
            b, t, _ = preds.shape
            preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
            preds = rearrange(preds, "(b t) d -> b t d", b=b, t=t)
            return preds

        raise ValueError(f"Unexpected predictor output rank: {preds.dim()}")

    def _action_encoder_input_dim(self):
        enc = getattr(self.action_encoder, "impl", self.action_encoder)
        encoder_action_dim = int(getattr(enc, "input_dim", 0) or 0)
        return encoder_action_dim

    @staticmethod
    def _parse_action_num_blocks_per_step(value):
        if value is None:
            return None
        if torch.is_tensor(value):
            value = value.detach().cpu().tolist()
        elif isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            if text[0] in "[(":
                value = ast.literal_eval(text)
            else:
                value = text.split(",")
        elif isinstance(value, (int, float)):
            value = [value]

        blocks = [int(v) for v in value]
        if len(blocks) == 0:
            return None
 
        return blocks

    def _rollout_consistency_weight(self, info_dict):
        weight = info_dict.get(
            "consistency_loss_weight",
            getattr(self, "consistency_loss_weight", 0.0),
            )
        if weight is None:
            weight = 0.0
        if torch.is_tensor(weight):
            weight = weight.detach().item()
        return float(weight)

    def _rollout_consistency_blocks(self, info_dict):
        return self._parse_action_num_blocks_per_step(
            info_dict.get(
                "action_num_blocks_per_step",
                getattr(self, "action_num_blocks_per_step", None),
            )
        )

    @staticmethod
    def prediction_loss(pred_emb, target_emb, *, supervise_last_state_only=False):
        """MSE prediction loss over all predicted states or only the final one."""
        if supervise_last_state_only:
            pred_emb = pred_emb[..., -1:, :]
            target_emb = target_emb[..., -1:, :]

        return (pred_emb - target_emb).pow(2).mean()

    ####################
    ## Inference only ##
    ####################
    
    def rollout(self, info, action_sequence):
   

        assert "pixels" in info, "pixels not in info_dict"
        return_intermediate_latents = bool(info.get("return_intermediate_latents", False))
        H = info["pixels"].size(2)
        B, S = action_sequence.shape[:2]
        
        encoder_action_dim = self._action_encoder_input_dim()
        packed_action_dim = int(action_sequence.size(-1))
      
        if packed_action_dim % encoder_action_dim != 0:
            raise ValueError(
                "Eval packed action dim must be divisible by action_encoder.input_dim. "
                f"packed={packed_action_dim}, encoder={encoder_action_dim}."
            )
        rollout_steps = action_sequence.size(2)
        action_num_blocks = packed_action_dim // encoder_action_dim

        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init.pop("action", None)
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        # Flatten (B, S) -> (B*S) so rollout runs as a single batched sequence.
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        action_blocks = rearrange(
            action_sequence,
            "b s t (f a) -> (b s) t f a",
            f=action_num_blocks,
            a=encoder_action_dim,
        )

       
        HS = 1
        initial_emb = emb
        intermediate_segments = [] if return_intermediate_latents else None
        if rollout_steps == 1:
            act_emb = self.action_encoder(
                action_blocks[:, 0],
                return_last_only=not return_intermediate_latents,
                latent=emb[:, -HS:],
            )
            emb_trunc = emb[:, -HS:]  # (BS, HS, D)
            pred_all = self.predict(emb_trunc, act_emb)
            if intermediate_segments is not None:
                intermediate_segments.append(pred_all)
            pred_emb = pred_all[:, -1:]  # (BS, 1, D)
            emb = torch.cat([emb, pred_emb], dim=1)
        else:
            for t in range(rollout_steps):
                act_emb = self.action_encoder(
                    action_blocks[:, t],
                    return_last_only=not return_intermediate_latents,
                    latent=emb[:, -HS:],
                )
                emb_trunc = emb[:, -HS:]  # (BS, HS, D)
                pred_all = self.predict(emb_trunc, act_emb)
                if intermediate_segments is not None:
                    intermediate_segments.append(pred_all)
                pred_emb = pred_all[:, -1:]  # (BS, 1, D)
                emb = torch.cat([emb, pred_emb], dim=1)

        # Restore separate batch and plan-sample dimensions.
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout
        if intermediate_segments is not None:
            dense_rollout = torch.cat([initial_emb, *intermediate_segments], dim=1)
            info["predicted_emb_trajectory"] = rearrange(
                dense_rollout, "(b s) ... -> b s ...", b=B, s=S
            )
        
        return info

    def rollout_action_num_blocks_per_step(
        self,
        initial_emb,
        action_sequence,
        action_num_blocks_per_step,
    ):
        B, S = action_sequence.shape[:2]
        encoder_action_dim = self._action_encoder_input_dim()
        packed_action_dim = int(action_sequence.size(-1))

        action_num_blocks = packed_action_dim // encoder_action_dim
        action_num_blocks_per_step = self._parse_action_num_blocks_per_step(
            action_num_blocks_per_step
        )
     
        rollout_steps = action_sequence.size(2)
        emb = rearrange(initial_emb, "b s ... -> (b s) ...").clone()
        action_blocks = rearrange(
            action_sequence,
            "b s t (f a) -> (b s) t f a",
            f=action_num_blocks,
            a=encoder_action_dim,
        )

        HS = 1
        step_end_embs = []
        for t in range(rollout_steps):
            block_start = 0
            for num_blocks in action_num_blocks_per_step:
                block_end = block_start + num_blocks
                act_emb = self.action_encoder(
                    action_blocks[:, t, block_start:block_end],
                    return_last_only=True,
                    latent=emb[:, -HS:],
                )
                emb_trunc = emb[:, -HS:]
                pred_all = self.predict(emb_trunc, act_emb)
                pred_emb = pred_all[:, -1:]
                emb = torch.cat([emb, pred_emb], dim=1)
                block_start = block_end
            step_end_embs.append(emb[:, -1:])

        step_end_embs = rearrange(
            torch.cat(step_end_embs, dim=1),
            "(b s) t d -> b s t d",
            b=B,
            s=S,
        )
        pred_rollout = torch.cat([initial_emb, step_end_embs], dim=2)
        return pred_rollout

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        pred_emb = info_dict["predicted_emb"]  # (B,S, T-1, dim)
        goal_emb = info_dict["goal_emb"]  # (B, S, T, dim)

        if goal_emb.dim() == pred_emb.dim() - 1:
            goal_emb = goal_emb.unsqueeze(1)
        if goal_emb.dim() != pred_emb.dim():
            raise ValueError(
                "Goal embedding rank must match predicted embedding rank "
                "or omit only the candidate-sample dimension. "
                f"pred={tuple(pred_emb.shape)}, goal={tuple(goal_emb.shape)}."
            )

        pred_last = pred_emb[..., -1:, :]
        goal_last = goal_emb[..., -1:, :].detach().expand_as(pred_last)

        # return last-step cost per action candidate
        cost = F.mse_loss(
            pred_last,
            goal_last,
            reduction="none",
        ).sum(dim=tuple(range(2, pred_last.ndim)))  # (B, S)

        consistency_weight = self._rollout_consistency_weight(info_dict)
        if consistency_weight != 0.0:
            consistency_pred = info_dict["consistency_predicted_emb"]
            consistency_last = consistency_pred[..., -1:, :]
            consistency_cost = F.mse_loss(
                pred_last,
                consistency_last,
                reduction="none",
            ).sum(dim=tuple(range(2, pred_last.ndim)))  # (B, S)
            info_dict["goal_cost"] = cost
            info_dict["consistency_cost"] = consistency_cost
            cost = cost + consistency_weight * consistency_cost

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """ Compute the cost of action candidates given an info dict with goal and initial state."""
        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        num_action_candidates = int(action_candidates.size(1))
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                value = info_dict[k]
                if value.dim() >= 2 and value.size(1) == num_action_candidates:
                    value = value[:, :1]
                info_dict[k] = value.to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]

        for k in info_dict:
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)

        goal.pop("action")
        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"]
        info_dict = self.rollout(info_dict, action_candidates)
        consistency_weight = self._rollout_consistency_weight(info_dict)
        consistency_blocks = self._rollout_consistency_blocks(info_dict)
        if consistency_weight != 0.0:
            info_dict["consistency_predicted_emb"] = (
                self.rollout_action_num_blocks_per_step(
                    info_dict["predicted_emb"][..., :1, :],
                    action_candidates,
                    consistency_blocks,
                )
            )
        return self.criterion(info_dict)
