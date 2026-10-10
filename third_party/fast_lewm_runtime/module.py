import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import math
def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift



class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024, merge_time_into_batch=False):
        super().__init__()
        self.num_proj = num_proj
        self.merge_time_into_batch = merge_time_into_batch
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        if self.merge_time_into_batch:
            proj = rearrange(proj, "t b d -> 1 (t b) d")

        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean() # average over projections and time
    
class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ValueOnlyProjection(nn.Module):
    

    def __init__(self, dim, value_heads=8, value_dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = value_dim_head * value_heads
        project_out = not (value_heads == 1 and value_dim_head == dim)
        self.norm = nn.LayerNorm(dim)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        v = self.to_v(x)
        return self.to_out(v)






class ConditionalPointwiseBlock(nn.Module):
    """Pointwise predictor block with AdaLN-zero conditioning."""

    def __init__(
        self,
        dim,
        value_heads,
        value_dim_head,
        mlp_dim,
        dropout=0.0,
    ):
        super().__init__()

        self.value_proj = ValueOnlyProjection(
            dim,
            value_heads=value_heads,
            value_dim_head=value_dim_head,
            dropout=dropout,
        )
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_value, scale_value, gate_value, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_value * self.value_proj(
            modulate(self.norm1(x), shift_value, scale_value)
        )
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
        block_kwargs=None,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])
        block_kwargs = block_kwargs or {}

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout, **block_kwargs)
            )

    def forward(self, x, c=None):

        if hasattr(self, "input_proj"):
            x = self.input_proj(x)

        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)

        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x


class ConditionalTransitionStack(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        value_heads,
        value_dim_head,
        mlp_dim,
        dropout=0.0,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                ConditionalPointwiseBlock(
                    hidden_dim,
                    value_heads,
                    value_dim_head,
                    mlp_dim,
                    dropout,
                )
            )

    def forward(self, x, c):
        x = self.input_proj(x)
        c = self.cond_proj(c)

        for block in self.layers:
            x = block(x, c)
        x = self.norm(x)

        return self.output_proj(x)




class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)






class ActionPrefixEmbedder(nn.Module):
    """
    ActionPrefixEmbedder embedder.

    """

    def __init__(
        self,
        input_dim=10,
        smoothed_dim=32,
        emb_dim=64,
        mlp_scale=4,
        variable_frameskip=False,
        use_positional_encoding=True,
        temporal_mixer_type="transformer",
        rnn_layers=1,
        transformer_depth=2,
        transformer_heads=8,
        transformer_dim_head=64,
        transformer_mlp_dim=None,
        use_latent_condition=False,
        latent_condition_mode="prefix",
        use_learnable_latent_token=False,
        latent_dim=None,
        dropout=0.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.variable_frameskip = bool(variable_frameskip)
        self.use_positional_encoding = bool(use_positional_encoding)
        self.emb_dim = int(emb_dim)
        self.temporal_mixer_type = str(temporal_mixer_type).lower()
        self.rnn_layers = int(rnn_layers)
        self.use_latent_condition = bool(use_latent_condition)
        self.use_learnable_latent_token = bool(use_learnable_latent_token)
        if self.use_learnable_latent_token and not self.use_latent_condition:
            raise ValueError(
                "use_learnable_latent_token=True requires use_latent_condition=True."
            )
        self.latent_condition_mode = str(latent_condition_mode).lower()
       
        self.latent_dim = (
            int(latent_dim) if latent_dim is not None else int(emb_dim)
        )
        self.transformer_mlp_dim = (
            int(transformer_mlp_dim)
            if transformer_mlp_dim is not None
            else int(4 * emb_dim)
        )
        self.latent_proj = (
            nn.Sequential(
                nn.Linear(self.latent_dim, 4 * emb_dim),
                nn.GELU(),
                nn.Linear(4 * emb_dim, emb_dim),
            )
            if self.use_latent_condition
            else None
        )
        if self.use_learnable_latent_token:
            self.learnable_latent_token = nn.Parameter(
                torch.empty(1, 1, self.latent_dim)
            )
            nn.init.normal_(self.learnable_latent_token, std=0.02)
        else:
            self.register_parameter("learnable_latent_token", None)

        # Per-action projection
        self.token_proj = nn.Linear(self.input_dim, smoothed_dim)

        # Per-action MLP
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

       
        if self.temporal_mixer_type == "gru":
            self.temporal_mixer = nn.GRU(
                input_size=emb_dim,
                hidden_size=emb_dim,
                num_layers=self.rnn_layers,
                batch_first=True,
                dropout=dropout if self.rnn_layers > 1 else 0.0,
            )
        elif self.temporal_mixer_type in {"transformer"}:
          
            self.temporal_mixer = Transformer(
                input_dim=emb_dim,
                hidden_dim=emb_dim,
                output_dim=emb_dim,
                depth=int(transformer_depth),
                heads=int(transformer_heads),
                dim_head=int(transformer_dim_head),
                mlp_dim=self.transformer_mlp_dim,
                dropout=dropout,
                block_class=Block,
            )
        else:
            raise ValueError(
                f"Unknown temporal_mixer_type={temporal_mixer_type!r}. "
                "Supported: ['gru', 'causal_transformer']."
            )
     
        self.out_norm = nn.LayerNorm(emb_dim)

    @staticmethod
    def _sinusoidal_pos_encoding(length, dim, device, dtype):
        """
        Returns: (1, length, dim)
        """
        pos = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)  # (L, 1)
        pe = torch.zeros(length, dim, device=device, dtype=dtype)

        div_term = torch.exp(
            torch.arange(0, dim, 2, device=device, dtype=dtype)
            * (-math.log(10000.0) / dim)
        )

        pe[:, 0::2] = torch.sin(pos * div_term)

      
        cos_width = pe[:, 1::2].shape[1]
        if cos_width > 0:
            pe[:, 1::2] = torch.cos(pos * div_term[:cos_width])

        return pe.unsqueeze(0)  # (1, L, D)

    def forward(self, x, return_last_only: bool = False, latent=None):
        """
        x: (B, F, input_dim)
        """
        x = x.float()
        x = self.token_proj(x)  # (B, F, smoothed_dim)
        x = self.embed(x)  # (B, F, emb_dim)

        latent_token = None
        latent_condition = None
        gru_h0 = None
        if self.use_latent_condition:
            if getattr(self, "use_learnable_latent_token", False):
                latent0 = self.learnable_latent_token.expand(x.size(0), -1, -1)
            else:
                latent = latent.float()
                if latent.dim() == 2:
                    latent = latent.unsqueeze(1)
                latent0 = latent[:, :1]
            latent0_proj = self.latent_proj(latent0)
            if self.temporal_mixer_type == "gru":
                gru_h0 = latent0_proj[:, 0]
                gru_h0 = (
                    gru_h0.unsqueeze(0)
                    .expand(self.rnn_layers, -1, -1)
                    .contiguous()
                )
            else:
                latent_token = latent0_proj
                latent_condition = None
                

        has_latent_prefix = (
            self.temporal_mixer_type != "gru" and latent_token is not None
        )
        if has_latent_prefix:
            x = torch.cat([latent_token, x], dim=1)

        if self.use_positional_encoding:
            pos = self._sinusoidal_pos_encoding(
                length=x.size(1),
                dim=x.size(-1),
                device=x.device,
                dtype=x.dtype,
            )
            x = x + pos

        # Keep all hidden states as prefix-action states.
        if self.temporal_mixer_type == "gru":
            x, _ = self.temporal_mixer(x, gru_h0)  # (B, F, emb_dim)
        else:
            # Attention module uses causal masking by default.
            x = self.temporal_mixer(
                x, c=latent_condition
            )  # (B, F, emb_dim) or (B, F + 1, emb_dim)
            if has_latent_prefix:
                x = x[:, 1:]  # drop z0 prefix; keep one output per action token.
      
        x = self.out_norm(x)
        if return_last_only:
            return x[:, -1:, :]
        return x


class Embedder(nn.Module):
   

    def __init__(
        self,
        *,
        embed_type: str = "action_prefix",
        **kwargs,
    ):
        super().__init__()
        self.embed_type = str(embed_type).lower()
        self.impl = ActionPrefixEmbedder(**kwargs)
       
    def forward(self, x, **kwargs):
        return self.impl(x, **kwargs)



class ARPredictor(nn.Module):
    """Autoregressive pointwise predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        depth,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        value_heads=None,
        value_dim_head=None,
        heads=None,
        dim_head=None,
        dropout=0.0,
        emb_dropout=0.0,
     
        token_processing="batch",
        action_fusion="residual_mlp",
        action_fusion_hidden_dim=None,
        action_fusion_zero_init=True,
    ):
        super().__init__()
        self.dropout = nn.Dropout(emb_dropout)
        self.input_dim = int(input_dim)
        
        self.token_processing = str(token_processing).lower()
        self.action_fusion = str(action_fusion).lower()
        if value_heads is None:
            value_heads = heads
      
        if value_dim_head is None:
            value_dim_head = dim_head if dim_head is not None else 64
        
        self.value_heads = int(value_heads)
        self.value_dim_head = int(value_dim_head)
    
   
       
      
        if self.action_fusion in {"none", "off", "disabled"}:
            self.action_fusion_norm = None
            self.action_fusion_mlp = None
        else:
            fusion_hidden_dim = (
                int(action_fusion_hidden_dim)
                if action_fusion_hidden_dim is not None
                else max(int(hidden_dim), 4 * int(input_dim))
            )
            self.action_fusion_norm = nn.LayerNorm(3 * int(input_dim))
            self.action_fusion_mlp = nn.Sequential(
                nn.Linear(3 * int(input_dim), fusion_hidden_dim),
                nn.GELU(),
                nn.Linear(fusion_hidden_dim, int(input_dim)),
            )
            if action_fusion_zero_init:
                nn.init.constant_(self.action_fusion_mlp[-1].weight, 0)
                nn.init.constant_(self.action_fusion_mlp[-1].bias, 0)
        self.transition_stack = ConditionalTransitionStack(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            self.value_heads,
            self.value_dim_head,
            mlp_dim,
            dropout,
        )

    def forward(self, x, c):
        """
        x:
          - (B, d)
          - (B, T, d)
        c:
          - (B, F, act_dim)
          - (B, T, act_dim)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
       

    
        # x is a single conditioning latent (B, 1, D),
        # c is prefix-action states (B, F, D).
        if x.size(1) == 1 and c.size(1) > 1:
            x = x.expand(-1, c.size(1), -1)
        
        x = self.dropout(x)
        if self.action_fusion_mlp is not None:
            
            fusion_input = torch.cat([x, c, x * c], dim=-1)
            x = x + self.action_fusion_mlp(
                self.action_fusion_norm(fusion_input)
            )
        bsz, tok_len, _ = x.shape
        x = rearrange(x, "b t d -> (b t) 1 d")
        c = rearrange(c, "b t d -> (b t) 1 d")
        out = self.transition_stack(x, c)  # (B*T, 1, D)
        return rearrange(out, "(b t) 1 d -> b t d", b=bsz, t=tok_len)

       