"""
Contiguous slice masking for Masked Slice Prediction (MSP).

Generates contiguous block masks analogous to V-JEPA's spatiotemporal tube
masking, adapted for 1-D slice sequences. Contiguous masking forces the model
to predict anatomically coherent regions rather than interpolating isolated
missing slices.

References:
    [1] Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding
        Predictive Architecture", CVPR 2023. https://arxiv.org/abs/2301.08243
    [2] Bardes et al., "Revisiting Feature Prediction for Learning Visual
        Representations from Video", ECCV 2024. https://arxiv.org/abs/2404.08471
    [3] Assran et al., "V-JEPA 2: Self-Supervised Video Models Enable
        Understanding, Prediction and Planning", 2025. https://arxiv.org/abs/2506.09985
"""

import random
import torch


def generate_contiguous_slice_mask(
    seq_lengths: torch.Tensor,
    mask_ratio_range: tuple[float, float] = (0.3, 0.5),
    max_seq_len: int | None = None,
) -> torch.Tensor:
    """
    Generate per-sample contiguous block masks for padded slice sequences.

    For each sample, a random contiguous block of slices (within the valid
    region defined by seq_lengths) is marked as masked. The block size is
    sampled uniformly from [min_ratio * L, max_ratio * L] where L is the
    valid sequence length of the sample.

    Args:
        seq_lengths: Actual (unpadded) sequence lengths per sample [B].
        mask_ratio_range: (min_ratio, max_ratio) controlling the fraction
            of valid slices to mask.
        max_seq_len: Temporal extent of the padded tensor.  When None the
            maximum value of seq_lengths is used.

    Returns:
        Boolean mask [B, max_seq_len] where True marks masked positions.
    """
    B = seq_lengths.shape[0]
    if max_seq_len is None:
        max_seq_len = seq_lengths.max().item()

    mask = torch.zeros(B, max_seq_len, dtype=torch.bool)
    min_ratio, max_ratio = mask_ratio_range

    for i in range(B):
        seq_len = seq_lengths[i].item()
        if seq_len < 2:
            continue
        lo = max(1, int(seq_len * min_ratio))
        hi = max(lo, int(seq_len * max_ratio))
        mask_len = random.randint(lo, hi)
        max_start = seq_len - mask_len
        start = random.randint(0, max(0, max_start))
        mask[i, start : start + mask_len] = True

    return mask


def generate_contiguous_slice_mask_packed(
    cu_seqlens: torch.Tensor,
    mask_ratio_range: tuple[float, float] = (0.3, 0.5),
) -> torch.Tensor:
    """
    Generate per-sample contiguous block masks for packed slice sequences.

    Operates on the flattened token dimension used by packed-sequence
    collation.  Each sample's valid region is delimited by cu_seqlens.

    Args:
        cu_seqlens: Cumulative sequence lengths [B+1] (int32).
        mask_ratio_range: (min_ratio, max_ratio) controlling the fraction
            of valid slices to mask per sample.

    Returns:
        Boolean mask [total_seq_len] where True marks masked positions.
    """
    total_seq_len = cu_seqlens[-1].item()
    B = cu_seqlens.shape[0] - 1
    mask = torch.zeros(total_seq_len, dtype=torch.bool)
    min_ratio, max_ratio = mask_ratio_range

    for i in range(B):
        start_idx = cu_seqlens[i].item()
        end_idx = cu_seqlens[i + 1].item()
        seq_len = end_idx - start_idx
        if seq_len < 2:
            continue
        lo = max(1, int(seq_len * min_ratio))
        hi = max(lo, int(seq_len * max_ratio))
        mask_len = random.randint(lo, hi)
        max_start = seq_len - mask_len
        start = random.randint(0, max(0, max_start))
        mask[start_idx + start : start_idx + start + mask_len] = True

    return mask
