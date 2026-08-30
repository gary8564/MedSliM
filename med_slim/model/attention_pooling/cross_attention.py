"""
Cross-attention pooling modules for inter-slice aggregation which aggregates a sequence of slice embeddings
[B, num_slices, embed_dim] into a volume embedding [B, embed_dim] using learnable query tokens.
"""

from typing import Sequence

import torch
import torch.nn as nn

ATTENTION_BLOCKS = ("self", "cross")
ALLOWED_ATTENTION_SCHEDULES = (("cross",), ("self", "cross"))


def validate_attention_blocks(attention_blocks: Sequence[str]) -> tuple[str, ...]:
    """
    Validate an attention block schedule and return it as a tuple.

    Allowed schedules are unique block types in a fixed order:
    ``["cross"]`` (query pooling) or ``["self", "cross"]`` (tokens mix, then
    learned queries read them). Cross-attention is required: this module pools
    with learned queries, so self-attention alone (then a mean) is not a
    supported aggregator.

    Duplicates are rejected: repeating a name would re-apply the same module,
    not add a new layer. ``["cross", "self"]`` is rejected because cross-attention
    already collapses the sequence to ``num_queries`` vectors (usually 1).
    """
    if isinstance(attention_blocks, str) or not isinstance(attention_blocks, (list, tuple)):
        raise ValueError(
            f"attention_blocks must be a list of {list(ATTENTION_BLOCKS)} entries, "
            f"got {attention_blocks!r} of type {type(attention_blocks).__name__}."
        )
    blocks = tuple(attention_blocks)
    if not blocks:
        raise ValueError(
            "attention_blocks must not be empty. "
            f"Allowed schedules: {[list(s) for s in ALLOWED_ATTENTION_SCHEDULES]}."
        )
    for block in blocks:
        if not isinstance(block, str) or block not in ATTENTION_BLOCKS:
            raise ValueError(
                f"Unknown attention block {block!r}. Only {list(ATTENTION_BLOCKS)} are allowed."
            )
    duplicates = sorted({b for b in blocks if blocks.count(b) > 1})
    if duplicates:
        raise ValueError(
            f"attention_blocks must not repeat entries, got duplicates {duplicates} "
            f"in {list(blocks)}. Repeating a name reuses the same weights; it is not "
            "extra depth. Allowed schedules: "
            f"{[list(s) for s in ALLOWED_ATTENTION_SCHEDULES]}."
        )
    if blocks not in ALLOWED_ATTENTION_SCHEDULES:
        raise ValueError(
            f"Unsupported attention_blocks {list(blocks)}. "
            f"Allowed schedules: {[list(s) for s in ALLOWED_ATTENTION_SCHEDULES]}."
        )
    return blocks


class _AttentionBlock(nn.Module):
    """Multi-head attention with an optional query residual and LayerNorm."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        use_residual: bool = True,
        use_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.use_residual = use_residual
        self.use_norm = use_norm
        self.mha = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        if use_norm:
            self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        out, weights = self.mha(
            query,
            keys,
            keys,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            average_attn_weights=True,
        )
        if self.use_residual:
            out = query + out
        if self.use_norm:
            out = self.norm(out)
        return out, weights


class InterSliceAggregator(nn.Module):
    """
    Inter-slice attention pooling with learnable query tokens.

    This module serves the same volume-level aggregation role as ABMIL:
    it pools a sequence of slice hidden states into a single volume embedding.
    The learnable query tokens attend to all tokens and the query outputs are
    avg-pooled. 

    `attention_blocks` is an ordered schedule. 
    The default `["cross"]` is query pooling.
    `["self", "cross"]` lets tokens interact before the queries read them, but self-attention scales quadratically in sequence length.

    Args:
        embed_dim:        Dimension of input hidden states.
        num_heads:        Number of attention heads.
        num_queries:      Number of learnable query vectors. Each attends
                          independently; outputs are avg-pooled to [B, embed_dim].
        use_residual:     Add the query residual after attention.
        use_norm:         Apply LayerNorm after the residual.
        dropout:          Dropout probability on attention weights.
        attention_blocks: ["cross"] or ["self", "cross"].
        query_init_std:   Std of the learned-query initialization.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_queries: int = 1,
        use_residual: bool = True,
        use_norm: bool = True,
        dropout: float = 0.0,
        attention_blocks: Sequence[str] = ("cross",),
        query_init_std: float = 0.02,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_queries = num_queries
        self.use_residual = use_residual
        self.use_norm = use_norm
        self.attention_blocks = validate_attention_blocks(attention_blocks)

        block_kwargs = dict(
            embed_dim=embed_dim,
            num_heads=num_heads,
            use_residual=use_residual,
            use_norm=use_norm,
            dropout=dropout,
        )
        if "self" in self.attention_blocks:
            self.self_attention = _AttentionBlock(**block_kwargs)
        if "cross" in self.attention_blocks:
            self.cross_attention = _AttentionBlock(**block_kwargs)
            self.learned_queries = nn.Parameter(
                torch.randn(num_queries, embed_dim) * query_init_std
            )

    def forward(
        self,
        h: torch.Tensor,
        mask: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h:               Token hidden states [B, num_tokens, embed_dim].
            mask:            Boolean mask [B, num_tokens], True = valid, False = padded.
            return_attention: If True, also return the last block's attention weights.

        Returns:
            pooled: Volume-level embedding [B, embed_dim].
            attn_weights (optional): [B, num_queries, num_tokens] for a schedule
                ending in cross-attention, averaged over heads.
        """
        key_padding_mask = ~mask if mask is not None else None
        x = h
        weights = None

        for block in self.attention_blocks:
            if block == "self":
                x, weights = self.self_attention(
                    x, x, key_padding_mask=key_padding_mask, need_weights=return_attention
                )
            else:
                queries = self.learned_queries.unsqueeze(0).expand(x.size(0), -1, -1)
                x, weights = self.cross_attention(
                    queries, x, key_padding_mask=key_padding_mask, need_weights=return_attention
                )
                # Padded tokens are consumed here; every query output is valid.
                key_padding_mask = None

        pooled = x.mean(dim=1)
        if return_attention:
            return pooled, weights
        return pooled
