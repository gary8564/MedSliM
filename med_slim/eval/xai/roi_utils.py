"""
Shared ROI annotation and class-label helpers for slice-level explanations.

Used by both the ABMIL attention CLI and the occlusion CLI so that "which slices
are the annotated lesion" and "is this a tear case" are answered identically.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional

from med_slim.utils.label_metadata import format_display_name


def load_roi_annotations(annotations_path: str) -> Optional[pd.DataFrame]:
    """Load ROI annotations when the CSV contains ROI columns."""
    df = pd.read_csv(annotations_path, dtype={"ID": str})
    if {"ID", "roiZ", "roiDepth"}.issubset(df.columns):
        return df.set_index("ID")
    return None


def discover_pathology_roi_columns(roi_df: Optional[pd.DataFrame]) -> list[str]:
    """Return pathology names that have ``roiZ_{name}`` / ``roiDepth_{name}`` columns."""
    if roi_df is None:
        return []
    prefixes = []
    for col in roi_df.columns:
        if col.startswith("roiZ_"):
            name = col[len("roiZ_"):]
            if f"roiDepth_{name}" in roi_df.columns:
                prefixes.append(name)
    return sorted(prefixes)


def get_roi_slice_range(
    roi_df: Optional[pd.DataFrame],
    sample_id: str,
    seq_len: int,
    z_col: str = "roiZ",
    depth_col: str = "roiDepth",
) -> Optional[Dict[str, Any]]:
    """Return clipped ROI slice metadata for a sample if available."""
    if roi_df is None or sample_id not in roi_df.index:
        return None

    row = roi_df.loc[sample_id]
    if pd.isna(row[z_col]) or pd.isna(row[depth_col]):
        return None

    roi_z = int(row[z_col])
    roi_depth = int(row[depth_col])
    if roi_depth <= 0:
        return None

    start = max(0, roi_z)
    end = min(seq_len - 1, roi_z + roi_depth - 1)
    if start > end:
        return None

    return {
        "roi_z": roi_z,
        "roi_depth": roi_depth,
        "roi_start_slice": start,
        "roi_end_slice": end,
    }


def compute_roi_attention_metrics(
    attention_weights: np.ndarray,
    roi_range: Dict[str, Any],
    top_ks: tuple[int, ...] = (1, 3, 5),
) -> Dict[str, Any]:
    """Compute metrics from slice attention and ROI slice range."""
    attn = attention_weights.astype(np.float64)
    attn = attn / attn.sum()

    start = int(roi_range["roi_start_slice"])
    end = int(roi_range["roi_end_slice"])
    roi_mask = np.zeros(len(attn), dtype=bool)
    roi_mask[start:end + 1] = True

    mass_in_roi = float(attn[roi_mask].sum())
    random_mass_in_roi = float(roi_mask.mean())
    peak_idx = int(np.argmax(attn))
    peak_in_roi = bool(roi_mask[peak_idx])

    metrics: Dict[str, Any] = {
        "roi_z": roi_range["roi_z"],
        "roi_depth": roi_range["roi_depth"],
        "roi_start_slice": start,
        "roi_end_slice": end,
        "mass_in_roi": mass_in_roi,
        "random_mass_in_roi": random_mass_in_roi,
        "mass_gain_over_random": float(mass_in_roi - random_mass_in_roi),
        "peak_in_roi": int(peak_in_roi),
    }

    ranked = np.argsort(attn)[::-1]
    for k in top_ks:
        effective_k = min(k, len(attn))
        topk_hit = bool(np.any(roi_mask[ranked[:effective_k]]))
        metrics[f"top_{k}_hit"] = int(topk_hit)

    return metrics


def resolve_class_label(
    task: str,
    label,
    target_labels: list[str],
    display_map,
    multiclass_names,
) -> str:
    """Human-readable class name for a sample label."""
    if task == "multilabel":
        active = [
            format_display_name(target_labels[j], display_map)
            for j, v in enumerate(label)
            if v == 1
        ]
        return " + ".join(active) if active else "Normal"
    if task == "multiclass" and multiclass_names:
        return multiclass_names[int(label)]
    return format_display_name(target_labels[0], display_map) if int(label) == 1 else "Normal"


def is_positive_pathology_class(class_label: str) -> bool:
    """True for partial/complete (or binary) tear classes; False for Normal."""
    return "tear" in class_label.strip().lower()
