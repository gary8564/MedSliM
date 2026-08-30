"""
Occlusion explanations for slice bags.

Two experiments share the same drop primitive:

- ``roi_group_occlusion``: drop the annotated lesion window as a group and compare
  the score change against equally wide lesion-free windows and against keeping
  only the lesion. Answers "did the classifier use the ACL slices at all".
- ``leave_one_slice_out``: drop exactly one slice at a time, always starting from
  the full bag, producing one score per slice. This is the occlusion heatmap.

Sign convention throughout: ``delta = F_full - F_dropped``, so a positive delta
means the dropped slices were supporting the explained class.
"""

import logging
import numpy as np
from typing import Any, Dict, List, Optional, Sequence

import torch

from med_slim.eval.xai.drop import BagInputs, score_keep
from med_slim.logging.setup import init_logging

init_logging()
logger = logging.getLogger(__name__)


def _complement(seq_len: int, occluded: Sequence[int]) -> List[int]:
    occluded_set = {int(i) for i in occluded}
    return [i for i in range(seq_len) if i not in occluded_set]


def sample_control_windows(
    seq_len: int,
    roi_start: int,
    roi_end: int,
    n_windows: int,
    rng: np.random.Generator,
) -> List[List[int]]:
    """
    Draw contiguous windows as wide as the ROI to act as a drop-size control.

    Non-overlapping windows are preferred so the control cannot accidentally remove
    lesion slices; when the ROI is too wide for any disjoint window to exist we fall
    back to any window with a different start, and report that via the returned count.
    """
    width = roi_end - roi_start + 1
    if width >= seq_len:
        return []

    starts = list(range(0, seq_len - width + 1))
    disjoint = [s for s in starts if s + width - 1 < roi_start or s > roi_end]
    candidates = disjoint if disjoint else [s for s in starts if s != roi_start]
    if not candidates:
        return []

    replace = len(candidates) < n_windows
    chosen = rng.choice(candidates, size=n_windows, replace=replace)
    return [list(range(int(s), int(s) + width)) for s in chosen]


def roi_group_occlusion(
    model: torch.nn.Module,
    bag: BagInputs,
    roi_range: Dict[str, Any],
    class_index: int,
    rng: np.random.Generator,
    n_random: int = 5,
) -> Dict[str, Any]:
    """
    Compare dropping the ROI window against control windows and against its complement.

    Args:
        model: Full classifier (COBRA + head) in eval mode.
        bag: Unpadded single volume.
        roi_range: ``get_roi_slice_range`` output for this volume.
        class_index: Explained class (the ground-truth label).
        rng: Generator for the control windows.
        n_random: Number of control windows to average.

    Returns:
        Flat dict of full-bag scores and ``delta_{logit,prob}_{roi,random,complement}``.
        ``delta_prob_complement`` uses the ROI-only bag, so a small value means the
        lesion slices alone are enough to reproduce the prediction.
    """
    seq_len = bag.seq_len
    roi_start = int(roi_range["roi_start_slice"])
    roi_end = int(roi_range["roi_end_slice"])
    roi_slices = list(range(roi_start, roi_end + 1))

    full = score_keep(model, bag, range(seq_len), class_index)
    roi_dropped = score_keep(model, bag, _complement(seq_len, roi_slices), class_index)
    complement_dropped = score_keep(model, bag, roi_slices, class_index)

    control_windows = sample_control_windows(seq_len, roi_start, roi_end, n_random, rng)
    control_scores = [
        score_keep(model, bag, _complement(seq_len, window), class_index)
        for window in control_windows
    ]

    row: Dict[str, Any] = {
        "seq_len": seq_len,
        "roi_start_slice": roi_start,
        "roi_end_slice": roi_end,
        "roi_width": len(roi_slices),
        "full_logit": full["logit"],
        "full_prob": full["prob"],
        "delta_logit_roi": full["logit"] - roi_dropped["logit"],
        "delta_prob_roi": full["prob"] - roi_dropped["prob"],
        "delta_logit_complement": full["logit"] - complement_dropped["logit"],
        "delta_prob_complement": full["prob"] - complement_dropped["prob"],
        "n_control_windows": len(control_scores),
        "control_windows_disjoint": all(
            max(w) < roi_start or min(w) > roi_end for w in control_windows
        ) if control_windows else False,
    }
    if control_scores:
        row["delta_logit_random"] = float(
            np.mean([full["logit"] - s["logit"] for s in control_scores])
        )
        row["delta_prob_random"] = float(
            np.mean([full["prob"] - s["prob"] for s in control_scores])
        )
    else:
        row["delta_logit_random"] = float("nan")
        row["delta_prob_random"] = float("nan")
    row["roi_gain_over_random"] = row["delta_logit_roi"] - row["delta_logit_random"]
    return row


def leave_one_slice_out(
    model: torch.nn.Module,
    bag: BagInputs,
    class_index: int,
    full_scores: Optional[Dict[str, float]] = None,
) -> Dict[str, np.ndarray]:
    """
    Score every slice by dropping it alone from the full bag.

    Each forward is independent: the delta for slice 5 does not depend on whether
    slice 3 was dropped earlier, which is what makes the result a per-slice heatmap
    rather than a cumulative curve.

    Returns:
        ``{"delta_logit": [seq_len], "delta_prob": [seq_len], "full": {...}}``.
        Deltas may be negative when a slice argued against the explained class.
    """
    seq_len = bag.seq_len
    full = full_scores or score_keep(model, bag, range(seq_len), class_index)

    delta_logit = np.zeros(seq_len, dtype=np.float64)
    delta_prob = np.zeros(seq_len, dtype=np.float64)
    for t in range(seq_len):
        dropped = score_keep(model, bag, _complement(seq_len, [t]), class_index)
        delta_logit[t] = full["logit"] - dropped["logit"]
        delta_prob[t] = full["prob"] - dropped["prob"]

    return {"delta_logit": delta_logit, "delta_prob": delta_prob, "full": full}
