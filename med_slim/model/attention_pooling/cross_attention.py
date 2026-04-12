"""
Cross-attention pooling for slice-level aggregation.

Uses a learnable query token to attend to all slice hidden states
via multi-head cross-attention, producing a single volume-level
representation with inter-slice competition built into the softmax
over key similarities.

References:
    Perceiver (Jaegle et al., 2021) - https://arxiv.org/abs/2103.03206
    MST (Chen et al., 2024) - CLS-token readout via cross-attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossAttentionPooling(nn.Module):
    """
    Multi-head cross-attention pooling with a single learnable query.

    A learnable query Q attends to the sequence of hidden states (K, V)
    to produce a single pooled vector.  Unlike ABMIL where each slice is
    scored independently, the attention weights here arise from Q-K
    similarity across all slices jointly.

    Args:
        embed_dim: Dimension of input hidden states.
        num_heads: Number of attention heads.
        dropout: Dropout on attention weights.
    """

    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_v = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: Sequence hidden states [B, T, D].
            mask: Boolean mask [B, T] where True = valid, False = padded.
            return_attention: If True, also return attention weights.

        Returns:
            pooled: Volume-level embedding [B, D].
            attn_weights (optional): [B, 1, T] averaged over heads.
        """
        B, T, _ = h.shape

        q = self.query.expand(B, -1, -1)  # [B, 1, D]
        k = self.W_k(h)                    # [B, T, D]
        v = self.W_v(h)                    # [B, T, D]

        # Reshape to multi-head: [B, num_heads, seq_len, head_dim]
        q = q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores [B, num_heads, 1, T]
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if mask is not None:
            # mask: [B, T] → [B, 1, 1, T]
            attn_logits = attn_logits.masked_fill(
                ~mask[:, None, None, :], float("-inf")
            )

        attn_weights = F.softmax(attn_logits, dim=-1)  # [B, num_heads, 1, T]
        attn_weights = self.attn_drop(attn_weights)

        # Weighted sum [B, num_heads, 1, head_dim]
        out = torch.matmul(attn_weights, v)
        # Reshape back [B, D]
        out = out.transpose(1, 2).contiguous().view(B, self.embed_dim)
        out = self.out_proj(out)

        if return_attention:
            # Average attention across heads → [B, 1, T]
            avg_attn = attn_weights.mean(dim=1)  # [B, 1, T]
            return out, avg_attn

        return out
