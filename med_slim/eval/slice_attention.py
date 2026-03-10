"""
COBRA Slice Attention Visualization.

Extracts and visualizes slice-level attention weights from COBRA model to understand
which slices contribute most to the final prediction.

Usage:
    python -m med_slim.eval.slice_attention \
        --experiment-dir /path/to/experiment_output
        --num-samples 10
"""

import os
import argparse
import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, Optional
from torch.utils.data import DataLoader
from accelerate import Accelerator

from med_slim.data.feat_dataset import FeatClassificationDataset, linear_classifier_collate_fn
from med_slim.eval.load_cobra import load_pretrained_cobra, load_cobra_from_experiment
from med_slim.eval.extract_feats import get_volume_attention, get_volume_attention_per_head
from med_slim.utils.viz.attention import plot_attention_profile, plot_per_head_attention_profile, compute_attention_metrics
from med_slim.utils.label_metadata import get_dataset_metadata, get_annotation_paths_by_split, format_display_name, build_multiclass_label_names
from med_slim.logging.setup import init_logging

init_logging()
logger = logging.getLogger(__name__)


def load_roi_annotations(annotations_path: str) -> Optional[pd.DataFrame]:
    """Load ROI annotations when the CSV contains ROI columns."""
    df = pd.read_csv(annotations_path, dtype={"ID": str})
    if {"ID", "roiZ", "roiDepth"}.issubset(df.columns):
        return df.set_index("ID")
    return None


def get_roi_slice_range(
    roi_df: Optional[pd.DataFrame],
    sample_id: str,
    seq_len: int,
) -> Optional[Dict[str, Any]]:
    """Return clipped ROI slice metadata for a sample if available."""
    if roi_df is None or sample_id not in roi_df.index:
        return None

    row = roi_df.loc[sample_id]
    if pd.isna(row["roiZ"]) or pd.isna(row["roiDepth"]):
        return None

    roi_z = int(row["roiZ"])
    roi_depth = int(row["roiDepth"])
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

def main():
    parser = argparse.ArgumentParser(description="Visualize COBRA slice attention weights")
    parser.add_argument("--experiment-dir", type=str, default=None,
                        help="Path to a linear-probing experiment directory. Loads model, features, annotations, and labels from config.yml and ckpt/classifier.pt directly.")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Path to COBRA checkpoint")
    parser.add_argument("--feat-dir", type=str, default=None, help="Directory with precomputed slice features")
    parser.add_argument("--annotations-path", type=str, default=None, help="Path to annotations CSV")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for plots")
    parser.add_argument("--dataset-name", type=str, default=None,
                        help="Dataset name (e.g. MRNet, SKM-TEA, kneeMRI)")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--plane", type=str, default=None, help="View plane")
    parser.add_argument("--fm-model-names", type=str, default=None, help="FM model names (space-separated)")
    parser.add_argument("--target-labels", type=str, nargs="+", default=None,
                        help="Target label columns to override the ones specified in `eval_datasets.yaml`")
    parser.add_argument("--task", type=str, default=None, choices=["binary", "multiclass", "multilabel"],
                        help="Classification task type to override the one specified in `eval_datasets.yaml`")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=None, help="Max samples to visualize (None = all)")
    parser.add_argument("--fm-pooling", type=str, default=None, choices=["avg_pool", "attention"])
    parser.add_argument("--sequence-encoder", type=str, default=None, choices=["mamba2", "transformer"])
    parser.add_argument("--slice-pooling", type=str, default=None, choices=["abmil", "cls"])
    parser.add_argument("--per-head", action="store_true",
                        help="Visualize per-head attention profiles (ABMIL multi-head only)")
    
    args = parser.parse_args()
    accelerator = Accelerator()

    # Configuration
    if args.experiment_dir:
        cobra_model, exp_cfg = load_cobra_from_experiment(args.experiment_dir, accelerator)
        cobra_model = cobra_model.to(accelerator.device)
        cobra_model.eval()

        feat_cfg = exp_cfg.get("feat_dataset", {})
        feat_dir = args.feat_dir or feat_cfg.get("feat_dir")
        plane = args.plane or feat_cfg.get("plane", ["sagittal"])[0]
        model_names = args.fm_model_names.split() if args.fm_model_names else feat_cfg.get("model_name", ["dinov2"])
        if isinstance(model_names, str):
            model_names = [model_names]
        dataset_name = args.dataset_name or feat_cfg.get("dataset_name")
        output_dir = args.output_dir or os.path.join(args.experiment_dir, "slice_attention")
        slice_pooling = args.slice_pooling  # Inferred by load_cobra_from_experiment
        if not args.annotations_path:
            annotations_dir = exp_cfg.get("annotations_dir")
            if not annotations_dir:
                raise ValueError(
                    "Cannot determine annotations_dir from experiment config. "
                    "Pass --annotations-path explicitly."
                )
            task_for_annots = args.task or exp_cfg.get("task") or get_dataset_metadata(dataset_name)["task"]
            if task_for_annots is None:
                raise ValueError(
                        "Cannot infer task for annotation path resolution. "
                        "Pass --task explicitly."
                    )
            annot_files = get_annotation_paths_by_split(annotations_dir, task_for_annots, [args.split])
            annotations_path = str(annot_files[args.split])
        else:
            annotations_path = args.annotations_path
    else:
        if not args.checkpoint_path:
            parser.error("--checkpoint-path is required when --experiment-dir is not used")
        if not args.feat_dir:
            parser.error("--feat-dir is required when --experiment-dir is not used")
        if not args.annotations_path:
            parser.error("--annotations-path is required when --experiment-dir is not used")
        if not args.output_dir:
            parser.error("--output-dir is required when --experiment-dir is not used")

        feat_dir = args.feat_dir
        plane = args.plane or "sagittal"
        model_names = args.fm_model_names.split() if args.fm_model_names else ["dinov2"]
        dataset_name = args.dataset_name
        output_dir = args.output_dir
        annotations_path = args.annotations_path
        slice_pooling = args.slice_pooling or "abmil"

        # Load COBRA from pretrained checkpoint
        pretrain_config_path = Path(args.checkpoint_path).parent / "config.yaml"
        cobra_cfg = {}
        if pretrain_config_path.exists():
            with open(pretrain_config_path) as f:
                pretrain_cfg = yaml.safe_load(f)
            cobra_cfg = pretrain_cfg.get("model", {}).get("cobra", {})

        cobra_model = load_pretrained_cobra(
            checkpoint_path=args.checkpoint_path,
            accelerator=accelerator,
            model_config=cobra_cfg,
            encoder_type="momentum",
            fm_pooling=args.fm_pooling or "avg_pool",
            sequence_encoder=args.sequence_encoder or "mamba2",
            slice_pooling=slice_pooling,
        )
        cobra_model = cobra_model.to(accelerator.device)
        cobra_model.eval()

    # Validate: attention visualization requires ABMIL slice pooling
    if slice_pooling and slice_pooling != "abmil":
        raise ValueError("Attention visualization requires ABMIL slice pooling")

    os.makedirs(output_dir, exist_ok=True)

    ds_meta = None
    if dataset_name:
        ds_meta = get_dataset_metadata(dataset_name)
    else:
        raise ValueError("Cannot determine dataset name.")
    if ds_meta is None:
        raise ValueError("Cannot determine dataset metadata.")
    target_labels = args.target_labels or ds_meta["target_labels"]
    task = args.task or ds_meta["task"]
    roi_df = load_roi_annotations(annotations_path)

    display_map = ds_meta.get("label_display_names") if ds_meta else None
    multiclass_maps = ds_meta.get("multiclass_label_maps") if ds_meta else None
    multiclass_names = None
    if task == "multiclass":
        multiclass_names = build_multiclass_label_names(target_labels[0], multiclass_maps)

    if accelerator.is_main_process:
        logger.info("=" * 60)
        logger.info("COBRA Slice Attention Visualization")
        logger.info("=" * 60)
        if args.experiment_dir:
            logger.info(f"Experiment dir: {args.experiment_dir}")
        logger.info(f"Feature dir: {feat_dir}")
        logger.info(f"Plane: {plane}, Split: {args.split}")
        logger.info(f"Task: {task}, Target labels: {target_labels}")
        logger.info("=" * 60)

    dataset = FeatClassificationDataset(
        feat_dir=feat_dir,
        slice_encoder_models=model_names,
        view_plane=plane,
        split=args.split,
        annotations_path=annotations_path,
        task=task,
        target_columns=target_labels,
    )

    if accelerator.is_main_process:
        logger.info(f"Loaded {len(dataset)} samples")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=linear_classifier_collate_fn,
        num_workers=4,
    )

    cobra_model, dataloader = accelerator.prepare(cobra_model, dataloader)

    # Extract attention weights
    attention_weights, labels, sample_ids, seq_lengths = get_volume_attention(
        cobra_model, dataloader, accelerator, max_samples=args.num_samples
    )

    # Extract per-head attention
    per_head_data = None
    effective_slice_pooling = slice_pooling or "abmil"
    if args.per_head and effective_slice_pooling == "abmil":
        per_head_data = get_volume_attention_per_head(
            cobra_model, dataloader, accelerator, max_samples=args.num_samples
        )

    if not accelerator.is_main_process:
        return

    logger.info(f"Extracted attention for {len(attention_weights)} samples")

    # Create output directories
    fig_dir = os.path.join(output_dir, "attention_profiles")
    os.makedirs(fig_dir, exist_ok=True)

    # Process each sample: plot + compute metrics
    all_metrics = []

    for attn, label, sample_id in zip(attention_weights, labels, sample_ids):
        attn_sum = float(attn.sum())
        if not np.isclose(attn_sum, 1.0, atol=1e-3):
            raise AssertionError(
                f"ABMIL attention should sum to 1 for sample '{sample_id}', got {attn_sum:.6f}"
            )
        seq_len = len(attn)
        roi_range = get_roi_slice_range(roi_df, str(sample_id), seq_len)

        if task == "multilabel":
            active = [format_display_name(target_labels[j], display_map) for j, v in enumerate(label) if v == 1]
            class_label = " + ".join(active) if active else "Normal"
        elif task == "multiclass" and multiclass_names:
            class_label = multiclass_names[int(label)]
        else:
            class_label = format_display_name(target_labels[0], display_map) if label == 1 else "Normal"

        plot_attention_profile(
            attention_weights=attn,
            sample_id=sample_id,
            view_plane=plane,
            output_dir=fig_dir,
            ground_truth_range=(
                (roi_range["roi_start_slice"], roi_range["roi_end_slice"])
                if roi_range is not None else None
            ),
            class_label=class_label,
        )

        metrics = compute_attention_metrics(attn)
        metrics['sample_id'] = sample_id
        metrics['class_label'] = class_label
        if roi_range is not None:
            metrics.update(compute_roi_attention_metrics(attn, roi_range))
        all_metrics.append(metrics)

    # Per-head attention visualization
    if per_head_data is not None:
        ph_attn, ph_labels, ph_sample_ids, _, num_heads = per_head_data
        logger.info(f"Plotting per-head attention for {len(ph_attn)} samples ({num_heads} heads)")

        per_head_dir = os.path.join(output_dir, "per_head_attention")
        os.makedirs(per_head_dir, exist_ok=True)

        for attn, label, sample_id in zip(ph_attn, ph_labels, ph_sample_ids):
            head_sums = attn.sum(axis=-1)
            if not np.allclose(head_sums, 1.0, atol=1e-3):
                raise AssertionError(
                    f"Per-head ABMIL attention should sum to 1 for sample '{sample_id}', got {head_sums:.6f}"
                )

            if task == "multilabel":
                active = [format_display_name(target_labels[j], display_map) for j, v in enumerate(label) if v == 1]
                ph_class_label = " + ".join(active) if active else "Normal"
            elif task == "multiclass" and multiclass_names:
                ph_class_label = multiclass_names[int(label)]
            else:
                ph_class_label = format_display_name(target_labels[0], display_map) if label == 1 else "Normal"

            plot_per_head_attention_profile(
                attention_weights=attn,
                sample_id=sample_id,
                view_plane=plane,
                output_dir=per_head_dir,
                class_label=ph_class_label,
            )

    # Save raw attention data
    stats_path = os.path.join(output_dir, "attention_stats.npz")
    np.savez(
        stats_path,
        attention_weights=np.array(attention_weights, dtype=object),
        labels=np.array(labels),
        sample_ids=np.array(sample_ids, dtype=object),
        seq_lengths=np.array(seq_lengths),
        plane=plane,
    )
    logger.info(f"Saved attention statistics to {stats_path}")

    # Aggregate statistics
    peak_positions = [m['peak_position'] / m['num_slices'] for m in all_metrics]

    logger.info("=" * 60)
    logger.info("Attention Statistics Summary")
    logger.info("=" * 60)
    logger.info(f"Total samples: {len(all_metrics)}")
    logger.info(f"Mean relative peak position: {np.mean(peak_positions):.3f} ± {np.std(peak_positions):.3f}")
    logger.info(f"Peak attention range: [{np.min([m['peak_attention'] for m in all_metrics]):.3f}, "
                f"{np.max([m['peak_attention'] for m in all_metrics]):.3f}]")
    roi_metrics = [m for m in all_metrics if "mass_in_roi" in m]
    if roi_metrics:
        logger.info(f"ROI-aware samples: {len(roi_metrics)}")
        logger.info(
            f"Mean mass_in_roi: {np.mean([m['mass_in_roi'] for m in roi_metrics]):.3f} "
            f"(random baseline: {np.mean([m['random_mass_in_roi'] for m in roi_metrics]):.3f})"
        )
        logger.info(
            f"Peak-in-ROI hit rate: {np.mean([m['peak_in_roi'] for m in roi_metrics]):.3f}"
        )
        logger.info(
            f"Top-3 hit rate: {np.mean([m['top_3_hit'] for m in roi_metrics]):.3f}, "
            f"Top-5 hit rate: {np.mean([m['top_5_hit'] for m in roi_metrics]):.3f}"
        )

    # Save metrics to CSV
    metrics_df = pd.DataFrame(all_metrics)
    metrics_csv_path = os.path.join(output_dir, "attention_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    logger.info(f"Saved attention metrics to {metrics_csv_path}")

    logger.info("=" * 60)
    logger.info("Done!")


if __name__ == "__main__":
    main()
