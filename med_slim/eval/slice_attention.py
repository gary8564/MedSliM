"""
COBRA Slice Attention Visualization.

Extracts and visualizes slice-level attention weights from COBRA model to understand
which slices contribute most to the final prediction.

Usage:
    python -m med_slim.eval.slice_attention \
        --checkpoint-path /path/to/cobra/checkpoint.pth.tar \
        --feat-dir /path/to/precomputed/features \
        --annotations-path /path/to/annotations.csv \
        --output-dir ./outputs \
        --plane sagittal \
        --num-samples 10
"""

import os
import argparse
import yaml
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from accelerate import Accelerator

from med_slim.data.feat_dataset import FeatClassificationDataset, linear_classifier_collate_fn
from med_slim.eval.load_cobra import load_pretrained_cobra
from med_slim.eval.extract_feats import get_volume_attention, get_volume_attention_per_head
from med_slim.utils.viz.attention import plot_attention_profile, plot_per_head_attention_profile, compute_attention_metrics
from med_slim.logging.setup import init_logging

init_logging()
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Visualize COBRA slice attention weights")
    
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to COBRA checkpoint")
    parser.add_argument("--feat-dir", type=str, required=True, help="Directory with precomputed slice features")
    parser.add_argument("--annotations-path", type=str, required=True, help="Path to annotations CSV")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for plots")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--plane", type=str, default="sagittal", help="View plane")
    parser.add_argument("--fm-model-names", type=str, default="dinov2", help="FM model names (space-separated)")
    parser.add_argument("--target-labels", type=str, nargs="+", default=["Abnormal"], help="Target label columns")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=None, help="Max samples to visualize (None = all)")
    parser.add_argument("--fm-pooling", type=str, default="avg_pool", choices=["avg_pool", "attention"])
    parser.add_argument("--sequence-encoder", type=str, default="mamba2", choices=["mamba2", "transformer"])
    parser.add_argument("--slice-pooling", type=str, default="abmil", choices=["abmil", "cls"])
    parser.add_argument("--per-head", action="store_true",
                        help="Visualize per-head attention profiles (ABMIL multi-head only)")
    
    args = parser.parse_args()
    
    # Validate: attention visualization requires ABMIL slice pooling
    if args.slice_pooling != "abmil":
        raise ValueError("Attention visualization requires --slice-pooling abmil")
    
    accelerator = Accelerator()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load pretrain config
    pretrain_config_path = Path(args.checkpoint_path).parent / "config.yaml"
    cobra_cfg = {}
    if pretrain_config_path.exists():
        with open(pretrain_config_path) as f:
            pretrain_cfg = yaml.safe_load(f)
        cobra_cfg = pretrain_cfg.get("model", {}).get("cobra", {})
    
    if accelerator.is_main_process:
        logger.info("=" * 60)
        logger.info("COBRA Slice Attention Visualization")
        logger.info("=" * 60)
        logger.info(f"Checkpoint: {args.checkpoint_path}")
        logger.info(f"Feature dir: {args.feat_dir}")
        logger.info(f"Plane: {args.plane}, Split: {args.split}")
        logger.info(f"Slice pooling: {args.slice_pooling}")
        logger.info("=" * 60)
    
    # Load COBRA model
    cobra_model = load_pretrained_cobra(
        checkpoint_path=args.checkpoint_path,
        accelerator=accelerator,
        model_config=cobra_cfg,
        encoder_type="momentum",
        fm_pooling=args.fm_pooling,
        sequence_encoder=args.sequence_encoder,
        slice_pooling=args.slice_pooling,
    )
    cobra_model = cobra_model.to(accelerator.device)
    cobra_model.eval()
    
    # Create dataset
    task = "multilabel" if len(args.target_labels) > 1 else "binary"
    model_names = args.fm_model_names.split()
    
    dataset = FeatClassificationDataset(
        feat_dir=args.feat_dir,
        slice_encoder_models=model_names,
        view_plane=args.plane,
        split=args.split,
        annotations_path=args.annotations_path,
        task=task,
        target_columns=args.target_labels,
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
    
    # Extract attention weights using shared function from extract_feats.py
    attention_weights, labels, sample_ids, seq_lengths = get_volume_attention(
        cobra_model, dataloader, accelerator, max_samples=args.num_samples
    )
    
    # Extract per-head attention before filtering to main process
    per_head_data = None
    if args.per_head and args.slice_pooling == "abmil":
        per_head_data = get_volume_attention_per_head(
            cobra_model, dataloader, accelerator, max_samples=args.num_samples
        )
    
    if not accelerator.is_main_process:
        return
    
    logger.info(f"Extracted attention for {len(attention_weights)} samples")
    
    # Create output directories
    fig_dir = os.path.join(args.output_dir, "attention_profiles")
    os.makedirs(fig_dir, exist_ok=True)
    
    # Process each sample: plot + compute metrics
    all_metrics = []
    
    for attn, label, sample_id in zip(attention_weights, labels, sample_ids):
        # Determine class label string for plot title
        if task == "multilabel":
            active_labels = [args.target_labels[j] for j, v in enumerate(label) if v == 1]
            class_label = " + ".join(active_labels) if active_labels else "Normal"
        else:
            class_label = args.target_labels[0] if label == 1 else "Normal"
        
        # Plot attention profile
        plot_attention_profile(
            attention_weights=attn,
            sample_id=sample_id,
            view_plane=args.plane,
            output_dir=fig_dir,
            class_label=class_label,
        )
        
        # Compute metrics using consolidated function
        metrics = compute_attention_metrics(attn)
        metrics['sample_id'] = sample_id
        metrics['class_label'] = class_label
        all_metrics.append(metrics)
    
    # Per-head attention visualization
    if per_head_data is not None:
        ph_attn, ph_labels, ph_sample_ids, _, num_heads = per_head_data
        logger.info(f"Plotting per-head attention for {len(ph_attn)} samples ({num_heads} heads)")
        
        per_head_dir = os.path.join(args.output_dir, "per_head_attention")
        os.makedirs(per_head_dir, exist_ok=True)
        
        for attn, label, sample_id in zip(ph_attn, ph_labels, ph_sample_ids):
            if task == "multilabel":
                active_labels = [args.target_labels[j] for j, v in enumerate(label) if v == 1]
                ph_class_label = " + ".join(active_labels) if active_labels else "Normal"
            else:
                ph_class_label = args.target_labels[0] if label == 1 else "Normal"
            
            plot_per_head_attention_profile(
                attention_weights=attn,
                sample_id=sample_id,
                view_plane=args.plane,
                output_dir=per_head_dir,
                class_label=ph_class_label,
            )
    
    # Save raw attention data
    stats_path = os.path.join(args.output_dir, "attention_stats.npz")
    np.savez(
        stats_path,
        attention_weights=np.array(attention_weights, dtype=object),
        labels=np.array(labels),
        sample_ids=np.array(sample_ids, dtype=object),
        seq_lengths=np.array(seq_lengths),
        plane=args.plane,
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
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(all_metrics)
    metrics_csv_path = os.path.join(args.output_dir, "attention_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    logger.info(f"Saved attention metrics to {metrics_csv_path}")
    
    logger.info("=" * 60)
    logger.info("Done!")


if __name__ == "__main__":
    main()
