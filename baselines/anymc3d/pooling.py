"""
AnyMC3D query-based attention pooling.

a = softmax(H q / sqrt(d)); v = a^T H (Eq. 5 in arXiv:2512.12887).
This is permutation-invariant MIL pooling with one learned task query.
"""

from typing import Optional

import torch
import torch.nn as nn


def lengths_to_mask(
    seq_lengths: torch.Tensor,
    max_seq_len: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Boolean slice mask ``[B, S]`` with True = valid (non-padded)."""
    if seq_lengths.ndim != 1:
        raise ValueError(f"seq_lengths must be [B], got shape {tuple(seq_lengths.shape)}.")
    device = seq_lengths.device if device is None else device
    positions = torch.arange(max_seq_len, device=device)
    return positions.unsqueeze(0) < seq_lengths.to(device=device).unsqueeze(1)


def slice_embeddings(features) -> torch.Tensor:
    if isinstance(features, torch.Tensor):
        hidden = features
    elif isinstance(features, (list, tuple)):
        if len(features) != 1:
            raise ValueError(
                "AnyMC3D uses one frozen 2D FM. Pass a single model_name; "
                f"got {len(features)} feature tensors."
            )
        hidden = features[0]
    else:
        raise TypeError(
            f"features must be a tensor or a single-element list of tensors, "
            f"got {type(features).__name__}."
        )
    if hidden.ndim != 3:
        raise ValueError(
            "AnyMC3D expects global-only slice embeddings [B, S, d]. "
            "Tiled regional-token caches are a MedSliM ablation and are not "
            f"used for this baseline, got shape {tuple(hidden.shape)}."
        )
    return hidden


class TaskQueryPooling(nn.Module):
    """Learned task query pooling: ``a = softmax(H q / sqrt(d)); v = a^T H``."""

    def __init__(self, embed_dim: int, query_init_std: float = 0.02):
        super().__init__()
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be > 0, got {embed_dim}.")
        self.embed_dim = int(embed_dim)
        self.query = nn.Parameter(torch.empty(self.embed_dim))
        bound = 2.0 * float(query_init_std)
        nn.init.trunc_normal_(
            self.query, mean=0.0, std=float(query_init_std), a=-bound, b=bound
        )

    def forward(
        self,
        hidden: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden: Slice embeddings ``[B, S, d]``.
            mask: Boolean mask ``[B, S]``, True = valid.
            return_attention: If True, also return ``a`` with shape ``[B, S]``.

        Returns:
            pooled: Volume embedding ``[B, d]``.
            attn (optional): Slice weights ``[B, S]``.
        """
        if hidden.ndim != 3 or hidden.size(-1) != self.embed_dim:
            raise ValueError(
                f"hidden must be [B, S, {self.embed_dim}], got {tuple(hidden.shape)}."
            )
        scale = self.embed_dim ** 0.5
        scores = torch.matmul(hidden, self.query) / scale
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        pooled = torch.bmm(attn.unsqueeze(1), hidden).squeeze(1)
        if return_attention:
            return pooled, attn
        return pooled
