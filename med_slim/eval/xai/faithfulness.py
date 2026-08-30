"""
MoRF (most-relevant-first) insertion / deletion faithfulness for slice heatmaps.

Port of xMIL's ``patch_drop_or_add`` to slice bags. It consumes a ranking 
(occlusion deltas, ABMIL attention, or a random permutation) and perturbs slices cumulatively in that order, 
unlike leave-one-slice-out where every forward restarts from the full bag.

- Deletion: step ``k`` scores the bag without the ``k`` highest-ranked slices. A faithful ranking makes ``p(true class)`` crash early, so lower AUPC is better.
- Insertion: step ``k`` scores the bag containing only the ``k`` highest-ranked slices. A faithful ranking makes ``p`` recover early, so higher AUPC is better.

Both curves are reported against the fraction of slices flipped so bags of different lengths can be averaged.
"""

import logging
import numpy as np
from typing import Dict, List, Sequence

import torch

from med_slim.eval.xai.drop import BagInputs, score_keep
from med_slim.logging.setup import init_logging

init_logging()
logger = logging.getLogger(__name__)

MORF_MODES = ("drop", "add")


def morf_order(scores: Sequence[float]) -> np.ndarray:
    """
    Slice indices sorted most-relevant-first (descending score).

    Ties break by ascending slice index so a constant heatmap gives a reproducible
    order instead of one that depends on the sort implementation.
    """
    scores = np.asarray(scores, dtype=np.float64)
    return np.lexsort((np.arange(scores.shape[0]), -scores))


def morf_curve(
    model: torch.nn.Module,
    bag: BagInputs,
    scores: Sequence[float],
    class_index: int,
    mode: str = "drop",
) -> Dict[str, np.ndarray]:
    """
    Probability curve as slices are cumulatively removed from / added to the bag.

    Args:
        model: Full classifier (COBRA + head) in eval mode.
        bag: Unpadded single volume.
        scores: Per-slice relevance, length ``seq_len``.
        class_index: Explained class (the ground-truth label).
        mode: ``"drop"`` for deletion, ``"add"`` for insertion.

    Returns:
        ``{"fractions": [seq_len+1], "probs": [seq_len+1], "logits": [seq_len+1],
        "order": [seq_len]}``. One slice is flipped per step, which is affordable at
        kneeMRI bag sizes (T ~ 25).
    """
    if mode not in MORF_MODES:
        raise ValueError(f"mode must be one of {MORF_MODES}, got '{mode}'.")
    seq_len = bag.seq_len
    if len(scores) != seq_len:
        raise ValueError(
            f"Expected {seq_len} slice scores to match the bag length, got {len(scores)}."
        )

    order = morf_order(scores)
    probs = np.zeros(seq_len + 1, dtype=np.float64)
    logits = np.zeros(seq_len + 1, dtype=np.float64)

    for k in range(seq_len + 1):
        top_k = order[:k]
        keep = [i for i in range(seq_len) if i not in set(top_k.tolist())] if mode == "drop" else top_k
        step = score_keep(model, bag, keep, class_index)
        probs[k] = step["prob"]
        logits[k] = step["logit"]

    return {
        "fractions": np.arange(seq_len + 1, dtype=np.float64) / seq_len,
        "probs": probs,
        "logits": logits,
        "order": order,
    }


def aupc(fractions: np.ndarray, probs: np.ndarray) -> float:
    """
    Area under the probability curve against the fraction of slices flipped.

    Normalized by the fraction axis (which always spans 0..1), so curves from bags
    of different lengths are directly comparable.
    """
    return float(np.trapezoid(probs, fractions))


def mean_curve(
    curves: List[Dict[str, np.ndarray]],
    num_points: int = 21,
) -> Dict[str, np.ndarray]:
    """
    Average curves from bags of different lengths on a common fraction grid.

    Bag lengths vary per volume, so the per-sample curves have different numbers of
    steps; each is linearly interpolated onto a shared 0..1 grid before averaging.
    """
    grid = np.linspace(0.0, 1.0, num_points)
    if not curves:
        return {"fractions": grid, "probs": np.full(num_points, np.nan)}
    stacked = np.stack(
        [np.interp(grid, c["fractions"], c["probs"]) for c in curves]
    )
    return {
        "fractions": grid,
        "probs": stacked.mean(axis=0),
        "probs_std": stacked.std(axis=0),
    }
