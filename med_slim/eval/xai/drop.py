"""
Bag slice-drop primitive for perturbation-based explanations.
Follows xMIL's ``perturbation_drop``: perturbed slices are removed from the bag and ``seq_lengths`` shrinks accordingly. 
"""

import logging
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, NamedTuple, Optional, Sequence

from med_slim.logging.setup import init_logging

init_logging()
logger = logging.getLogger(__name__)


@dataclass
class BagInputs:
    """A single unpadded volume: K FM tensors plus its bag length."""
    features: List[torch.Tensor]
    seq_lengths: torch.Tensor
    physical_positions: Optional[torch.Tensor]

    @property
    def seq_len(self) -> int:
        return int(self.seq_lengths[0].item())


class DroppedBag(NamedTuple):
    """Bag after dropping slices. ``is_empty`` marks the zero-bag fallback."""
    features: List[torch.Tensor]
    seq_lengths: torch.Tensor
    physical_positions: Optional[torch.Tensor]
    is_empty: bool


def bag_from_batch(batch: Dict, device: torch.device) -> BagInputs:
    """
    Extract the single volume of a ``batch_size=1`` batch, stripped of padding.

    Slicing off the padding here means every downstream ``keep_idx`` indexes the
    real sequence ``[0, seq_len)`` and no drop can accidentally retain a pad token.
    """
    seq_lengths = batch["seq_lengths"]
    if seq_lengths.shape[0] != 1:
        raise ValueError(
            f"Slice-drop explanations need batch_size=1, got batch of {seq_lengths.shape[0]}."
        )
    seq_len = int(seq_lengths[0].item())
    features = [f[:, :seq_len].to(device) for f in batch["features"]]
    physical_positions = batch.get("physical_positions")
    if physical_positions is not None:
        physical_positions = physical_positions[:, :seq_len].to(device)
    return BagInputs(
        features=features,
        seq_lengths=seq_lengths.to(device),
        physical_positions=physical_positions,
    )


def drop_slices(
    features: List[torch.Tensor],
    seq_lengths: torch.Tensor,
    physical_positions: Optional[torch.Tensor],
    keep_idx: Sequence[int],
) -> DroppedBag:
    """
    Keep only ``keep_idx`` slices of a single-volume bag; drop the rest.

    Args:
        features: List of K tensors, each ``[1, seq_len, ...]``.
        seq_lengths: ``[1]`` bag length.
        physical_positions: Optional ``[1, seq_len]``. Gathered alongside the
            features so the tensors stay aligned; this does not enable physical PE,
            which Cobra applies only when ``physical_pe=True``.
        keep_idx: Indices into the **unpadded** sequence ``[0, seq_len)``. Treated
            as a set: duplicates are removed and the kept slices stay in ascending
            anatomical order, since the sequence encoder is order sensitive.

    Returns:
        ``DroppedBag``. An empty keep set yields a one-token zero bag (xMIL's
        empty-bag baseline) with ``is_empty=True``, because the encoder cannot run
        on a zero-length sequence.
    """
    if seq_lengths.shape[0] != 1:
        raise ValueError(f"drop_slices expects a single volume, got batch of {seq_lengths.shape[0]}.")
    seq_len = int(seq_lengths[0].item())

    keep = sorted({int(i) for i in keep_idx})
    out_of_range = [i for i in keep if i < 0 or i >= seq_len]
    if out_of_range:
        raise IndexError(
            f"keep_idx {out_of_range} out of range for a bag of {seq_len} slices."
        )

    device = features[0].device
    if not keep:
        empty_features = [
            torch.zeros((1, 1, *f.shape[2:]), dtype=f.dtype, device=f.device)
            for f in features
        ]
        empty_positions = None
        if physical_positions is not None:
            empty_positions = torch.zeros(
                (1, 1), dtype=physical_positions.dtype, device=physical_positions.device
            )
        return DroppedBag(
            features=empty_features,
            seq_lengths=torch.ones(1, dtype=seq_lengths.dtype, device=device),
            physical_positions=empty_positions,
            is_empty=True,
        )

    index = torch.as_tensor(keep, dtype=torch.long, device=device)
    kept_features = [f.index_select(1, index.to(f.device)) for f in features]
    kept_positions = None
    if physical_positions is not None:
        kept_positions = physical_positions.index_select(1, index.to(physical_positions.device))
    return DroppedBag(
        features=kept_features,
        seq_lengths=torch.full((1,), len(keep), dtype=seq_lengths.dtype, device=device),
        physical_positions=kept_positions,
        is_empty=False,
    )


@torch.no_grad()
def predict_class_scores(
    model: torch.nn.Module,
    features: List[torch.Tensor],
    seq_lengths: torch.Tensor,
    physical_positions: Optional[torch.Tensor],
    class_index: int,
) -> Dict[str, float]:
    """
    Score one bag for a single explained class.

    Multiclass heads take ``logits[class_index]`` and its softmax probability.
    Binary heads emit one logit for the positive class, so ``class_index=0`` flips
    its sign (``log p0/p1 = -log p1/p0``).

    Returns:
        ``{"logit": float, "prob": float}``.
    """
    out = model(features, seq_lengths, physical_positions)
    logits = out["logits"][0].float()

    if logits.numel() == 1:
        logit = logits[0]
        prob = torch.sigmoid(logit)
        if class_index == 0:
            logit = -logit
            prob = 1.0 - prob
        elif class_index != 1:
            raise IndexError(
                f"class_index={class_index} is invalid for a single-logit binary head."
            )
    else:
        if not 0 <= class_index < logits.numel():
            raise IndexError(
                f"class_index={class_index} out of range for {logits.numel()} classes."
            )
        logit = logits[class_index]
        prob = F.softmax(logits, dim=-1)[class_index]

    return {"logit": float(logit.item()), "prob": float(prob.item())}


def score_keep(
    model: torch.nn.Module,
    bag: BagInputs,
    keep_idx: Sequence[int],
    class_index: int,
) -> Dict[str, float]:
    """Drop everything outside ``keep_idx`` and score the resulting bag."""
    dropped = drop_slices(bag.features, bag.seq_lengths, bag.physical_positions, keep_idx)
    scores = predict_class_scores(
        model, dropped.features, dropped.seq_lengths, dropped.physical_positions, class_index
    )
    scores["is_empty"] = dropped.is_empty
    scores["num_kept"] = 0 if dropped.is_empty else int(dropped.seq_lengths[0].item())
    return scores


@torch.no_grad()
def abmil_attention(model: torch.nn.Module, bag: BagInputs) -> np.ndarray:
    """
    Pooling attention ``A`` actually used by ``A @ raw``, as a ``[seq_len]`` array.

    This is the mean-over-heads-then-softmax map from ``Cobra._abmil_pooling``, not
    the per-head reduction used for attention plots, so MoRF rankings are compared
    against the map the classifier really pools with.
    """
    cobra = model.cobra
    dtype = next(cobra.parameters()).dtype
    features = [f.to(dtype=dtype) for f in bag.features]
    positions = bag.physical_positions
    if positions is not None:
        positions = positions.to(dtype=torch.float32)
    attention = cobra(
        features,
        seq_lengths=bag.seq_lengths,
        physical_positions=positions,
        fm_ids=getattr(model, "fm_ids", None),
        get_attention=True,
    )  # [1, 1, seq_len]
    attention = attention[0, 0, : bag.seq_len].float().cpu().numpy()
    return attention
