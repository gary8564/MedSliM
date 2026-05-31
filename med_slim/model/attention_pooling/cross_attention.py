"""
Cross-attention pooling modules for within-slice and inter-slice aggregation.

- ``InterSliceAggregator`` aggregates a sequence of slice embeddings
  ``[B, num_slices, embed_dim]`` into a volume embedding ``[B, embed_dim]`` using learnable query tokens.
- ``WithinSliceAggregator`` aggregates tiled regional tokens within each slice
  ``[B, num_slices, num_tiled_regions, embed_dim]`` into one slice representation
  ``[B, num_slices, embed_dim]`` using the global token as the query.
"""

import torch
import torch.nn as nn


class InterSliceAggregator(nn.Module):
    """
    Inter-slice cross-attention pooling with learnable query tokens.

    This module serves the same volume-level aggregation role as ABMIL: 
    it pools a sequence of slice hidden states into a single volume embedding. 
    The learnable query tokens attend to all slices and the query outputs are avg-pooled.

    References:
       Dancette et al., 2025 (https://arxiv.org/abs/2509.06830)

    Args:
        embed_dim:    Dimension of input hidden states.
        num_heads:    Number of attention heads.
        num_queries:  Number of learnable query vectors. Each attends
                      independently; outputs are avg-pooled to [B, embed_dim].
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
            h:               Slice hidden states [B, num_slices, embed_dim].
            mask:            Boolean mask [B, num_slices], True = valid, False = padded.
            return_attention: If True, also return attention weights.

        Returns:
            pooled: Volume-level embedding [B, embed_dim].
            attn_weights (optional): [B, num_queries, num_slices] averaged over heads.
        """
        B = h.size(0)
        q = self.learned_queries.unsqueeze(0).expand(B, -1, -1)  # [B, num_queries, embed_dim]

        key_padding_mask = ~mask if mask is not None else None

        attn_out, attn_weights = self.mha(
            q, h, h,
            key_padding_mask=key_padding_mask,
            need_weights=return_attention,
            average_attn_weights=True,  # average over heads
        ) # attn_out: [B, num_queries, embed_dim]

        if self.use_residual:
            attn_out = q + attn_out
        if self.use_norm:
            attn_out = self.norm(attn_out)

        # Avg-pool across queries to get [B, embed_dim]
        pooled = attn_out.mean(dim=1)

        if return_attention:
            return pooled, attn_weights  # [B, num_queries, num_slices]
        return pooled


class WithinSliceAggregator(nn.Module):
    """
    Hierarchical within-slice aggregation for tiled multi-crop CLS features.

    Collapses per-slice region tokens
    ``[B, num_slices, num_tiled_regions, embed_dim]`` (num_tiled_regions =
    1 global + regional crop tokens) into a single slice representation
    ``[B, num_slices, embed_dim]`` before the inter-slice
    sequence encoder. The global token queries the regional tokens via cross-attention
    and is kept as a residual anchor so whole-slice context is never discarded:

        slice_rep = LayerNorm(global_cls + attention(global_cls -> regional_tokens))

    Region attention weights are returned for quadrant-level interpretability.

    Args:
        embed_dim: Token embedding dimension.
        num_heads: Cross-attention heads. Falls back to 1 if ``embed_dim`` is not
            divisible by ``num_heads``.
        dropout: Attention dropout.
        num_regions: Total tokens per slice (global + regional). When provided, a small
            learned region embedding is added so the aggregator can use quadrant
            identity. When None, regions are treated as an unordered set.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1,
                 num_regions: int | None = None):
        super().__init__()
        if embed_dim % num_heads != 0:
            num_heads = 1
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, dropout=dropout
        )
        self.region_embed = (
            nn.Embedding(num_regions, embed_dim) if num_regions is not None else None
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tokens: [B, num_slices, num_tiled_regions, embed_dim] with token 0 =
                global CLS, tokens 1.. = regional CLS.

        Returns:
            slice_rep: [B, num_slices, embed_dim] aggregated per-slice representation.
            region_attn: [B, num_slices, 1, num_tiled_regions-1] global->region attention weights.
        """
        batch_size, num_slices, num_tiled_regions, embed_dim = tokens.shape
        if self.region_embed is not None:
            region_ids = torch.arange(num_tiled_regions, device=tokens.device)
            tokens = tokens + self.region_embed(region_ids).view(1, 1, num_tiled_regions, embed_dim)
        x = tokens.reshape(batch_size * num_slices, num_tiled_regions, embed_dim)
        query = x[:, 0:1]            # global CLS as query
        kv = x[:, 1:]                # regional CLS as keys/values
        attended, attn = self.cross_attn(query, kv, kv, need_weights=True)
        out = self.norm(query + attended).squeeze(1)        # residual global anchor
        return (
            out.reshape(batch_size, num_slices, embed_dim),
            attn.reshape(batch_size, num_slices, 1, num_tiled_regions - 1),
        )