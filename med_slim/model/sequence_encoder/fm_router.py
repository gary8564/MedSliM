"""
Soft FM embedding router.

The router scores each FM token independently with a shared scorer 
with the number of FMs being treated like a variable-length token/set axis. 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FMRouter(nn.Module):
    """
    Per-slice soft router over precomputedFM features.

    Args:
        embed_dim: Embedding dimension.
        num_fms: Total number of foundation models.
        hidden_dim: Hidden dimension of the scorer MLP. Defaults to ``embed_dim``.
        use_fm_embedding: Add a learned per-FM identity embedding to each FM token before scoring.
        use_fm_logit_bias: Add a learned per-FM additive logit bias to each FM token before scoring.
        use_fm_logit_scale: Multiply logits by a learned per-FM positive scale factor to each FM token before scoring.
        dropout: Dropout inside the scorer MLP.
        router_mode: "soft" (dense softmax) or "topk" (sparse top-k softmax).
        top_k: Number of FMs kept when router_mode="topk".
        temperature: Softmax temperature.
        learnable_temperature: If True, the temperature is a learned parameter.
    """

    def __init__(
        self,
        embed_dim: int,
        num_fms: int,
        hidden_dim: int | None = None,
        use_fm_embedding: bool = True,
        use_fm_logit_bias: bool = True,
        use_fm_logit_scale: bool = False,
        dropout: float = 0.0,
        router_mode: str = "soft",
        top_k: int | None = None,
        temperature: float = 1.0,
        learnable_temperature: bool = False,
    ):
        super().__init__()
        if router_mode not in ("soft", "topk"):
            raise ValueError(
                f"Invalid router_mode '{router_mode}'. Must be 'soft' or 'topk'."
            )
        if router_mode == "topk" and (top_k is None or top_k < 1):
            raise ValueError("router_mode='topk' requires top_k >= 1.")

        self.embed_dim = embed_dim
        self.num_fms = num_fms
        self.router_mode = router_mode
        self.top_k = top_k

        if learnable_temperature:
            self.log_temperature = nn.Parameter(
                torch.log(torch.tensor(float(temperature)))
            )
        else:
            self.register_buffer(
                "log_temperature", torch.log(torch.tensor(float(temperature)))
            )

        self.fm_embed = nn.Embedding(num_fms, embed_dim) if use_fm_embedding else None
        self.fm_logit_bias = nn.Embedding(num_fms, 1) if use_fm_logit_bias else None
        self.fm_logit_scale = nn.Embedding(num_fms, 1) if use_fm_logit_scale else None
        if self.fm_embed is not None:
            nn.init.normal_(self.fm_embed.weight, std=0.02)
        if self.fm_logit_bias is not None:
            nn.init.zeros_(self.fm_logit_bias.weight)
        if self.fm_logit_scale is not None:
            # softplus(0.5413) ~ 1.0 so per-FM scale starts close to identity.
            nn.init.constant_(self.fm_logit_scale.weight, 0.5413)

        hidden_dim = hidden_dim or embed_dim
        self.scorer = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        fm_embs: torch.Tensor,                  # [K, B, num_slices, E]
        fm_ids: torch.Tensor,                   # [B, K] global FM IDs
        return_stats: bool = False,
    ):
        """
        Fuse FM embeddings per slice with a shared variable-K scorer.

        Returns:
            fused: [B, num_slices, E].
            stats (optional): dict with fm_weights_local [B, num_slices, K], fm_ids [B, K], router_logits [B, num_slices, K] and fm_weight_entropy [B, num_slices].
        """
        fm_ids = fm_ids.to(device=fm_embs.device, dtype=torch.long)
        tokens = fm_embs.permute(1, 2, 0, 3)  # [B, num_slices, K, E]
        scorer_tokens = tokens

        if self.fm_embed is not None:
            # FM identity conditions routing scores only
            scorer_tokens = scorer_tokens + self.fm_embed(fm_ids)[:, None, :, :]

        logits = self.scorer(scorer_tokens).squeeze(-1)  # [B, num_slices, K]

        if self.fm_logit_scale is not None:
            scale = F.softplus(self.fm_logit_scale(fm_ids)).squeeze(-1)  # [B, K]
            logits = logits * scale[:, None, :]
        if self.fm_logit_bias is not None:
            logits = logits + self.fm_logit_bias(fm_ids).squeeze(-1)[:, None, :]

        raw_router_logits = logits

        # Keep router softmax in float32: 
        # MoE routers can be numerically fragile in low precision 
        temperature = self.log_temperature.exp().clamp_min(1e-6)
        logits = logits.float() / temperature

        if self.router_mode == "soft" or self.top_k is None:
            weights = torch.softmax(logits, dim=-1)
        else:
            k = min(self.top_k, logits.shape[-1])
            top_vals, top_idx = torch.topk(logits, k=k, dim=-1)
            sparse_logits = torch.full_like(logits, torch.finfo(logits.dtype).min)
            sparse_logits.scatter_(-1, top_idx, top_vals)
            weights = torch.softmax(sparse_logits, dim=-1)

        fused = (weights.unsqueeze(-1).to(tokens.dtype) * tokens).sum(dim=2)  # [B, num_slices, E]

        if not return_stats:
            return fused

        probs = weights.clamp_min(1e-8)
        stats = {
            "fm_weights_local": weights,        # [B, num_slices, K]
            "fm_ids": fm_ids,                   # [B, K]
            "router_logits": raw_router_logits, # [B, num_slices, K]
            "router_logits_scaled": logits,     # [B, num_slices, K]
            "fm_weight_entropy": -(probs * probs.log()).sum(dim=-1),  # [B, num_slices]
        }
        return fused, stats
