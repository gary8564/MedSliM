"""
Cross-attention pooling for slice-level aggregation.

Uses learnable query tokens to attend to all slice hidden states via
multi-head cross-attention, producing a single volume-level embedding.
The learned query tokens attend to the slice hidden states to learn which
slices carry the most diagnostic signal.

References:
   Dancette et al., 2025 (https://arxiv.org/abs/2509.06830)
"""

import torch
import torch.nn as nn


class CrossAttentionPooling(nn.Module):
    """
    Multi-head cross-attention pooling with learnable query tokens.

    Args:
        embed_dim:    Dimension of input hidden states.
        num_heads:    Number of attention heads.
        num_queries:  Number of learnable query vectors. Each attends
                      independently; outputs are mean-pooled to [B, D].
        use_residual: Add query residual after attention.
        use_norm:     Apply LayerNorm after the residual.
        dropout:      Dropout probability on attention weights.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_queries: int = 1,
        use_residual: bool = True,
        use_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.use_residual = use_residual
        self.use_norm = use_norm

        self.learned_queries = nn.Parameter(torch.randn(num_queries, embed_dim) * 0.02)
        self.mha = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        if use_norm:
            self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        h: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h:               Slice hidden states [B, T, D].
            mask:            Boolean mask [B, T], True = valid, False = padded.
            return_attention: If True, also return attention weights.

        Returns:
            pooled: Volume-level embedding [B, D].
            attn_weights (optional): [B, num_queries, T] averaged over heads.
        """
        B = h.size(0)
        q = self.learned_queries.unsqueeze(0).expand(B, -1, -1)  # [B, num_queries, D]

        key_padding_mask = ~mask if mask is not None else None

        attn_out, attn_weights = self.mha(
            q, h, h,
            key_padding_mask=key_padding_mask,
            need_weights=return_attention,
            average_attn_weights=True,  # average over heads
        ) # attn_out: [B, num_queries, D]

        if self.use_residual:
            attn_out = q + attn_out
        if self.use_norm:
            attn_out = self.norm(attn_out)

        # Avg-pool across queries to get [B, D]
        pooled = attn_out.mean(dim=1)

        if return_attention:
            return pooled, attn_weights  # [B, num_queries, T]
        return pooled
