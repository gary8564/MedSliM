"""
COBRA Embedding Extraction and UMAP Clustering.

Extracts stack-level or slice-level embeddings using COBRA and visualizes clustering.

Usage:
    # Aggregated embeddings - one point per volume
    python -m med_slim.eval.embed_cluster \
        --checkpoint-path /path/to/cobra/checkpoint.pth.tar \
        --feat-dir /path/to/precomputed/features \
        --annotations-path /path/to/annotations.csv \
        --output-dir ./outputs \
        --target-labels acl meniscus \
        --plane sagittal \
        --title "Aggregated Embedding Space"

    # Slice-level embeddings - one point per slice
    python -m med_slim.eval.embed_cluster \
        --checkpoint-path /path/to/checkpoint.pth.tar \
        --feat-dir /path/to/features \
        --annotations-path /path/to/annotations.csv \
        --output-dir ./outputs \
        --target-labels acl \
        --slice-level \
        --color-by slice_position \
        --title "Slice-Level Embedding Space"
"""

import os
import argparse
import yaml
import logging
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm

from med_slim.data.feat_dataset import FeatClassificationDataset, linear_classifier_collate_fn
from med_slim.eval.load_cobra import load_pretrained_cobra
from med_slim.utils.viz.cluster import plot_embedding_clustering
from med_slim.logging.setup import init_logging

init_logging()
logger = logging.getLogger(__name__)


def extract_embeddings(
    cobra_model,
    dataloader: DataLoader,
    accelerator: Accelerator,
    slice_level: bool = False,
):
    """
    Extract COBRA embeddings from a dataloader.
    
    Args:
        cobra_model: COBRA model
        dataloader: DataLoader with features
        accelerator: Accelerator for distributed training
        slice_level: If True, return slice-level embeddings before aggregation
    
    Returns:
        If slice_level=False:
            embeddings: [N, embed_dim] numpy array
            labels: [N] or [N, num_labels] numpy array
            sample_ids: List of sample IDs
            slice_positions: None
            volume_ids: None
        If slice_level=True:
            embeddings: [N * num_slices, embed_dim] numpy array
            labels: [N * num_slices] or [N * num_slices, num_labels] numpy array  
            sample_ids: List of sample IDs (repeated for each slice)
            slice_positions: [N * num_slices] numpy array of slice positions
            volume_ids: [N * num_slices] numpy array of volume indices
    """
    cobra_model.eval()
    all_embeddings = []
    all_labels = []
    all_sample_ids = []
    all_slice_positions = []
    all_volume_ids = []
    volume_counter = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings", disable=not accelerator.is_main_process):
            seq_lengths = batch["seq_lengths"].to(accelerator.device)
            features = [f.to(accelerator.device, dtype=next(cobra_model.parameters()).dtype) for f in batch["features"]]
            batch_labels = batch["labels"]
            batch_sample_ids = batch["sample_ids"]
            
            if slice_level:
                # Get slice-level embeddings [B, num_slices, embed_dim]
                embeddings = cobra_model(features, seq_lengths=seq_lengths, return_slice_embeddings=True)
                B, T, E = embeddings.shape
                
                # Create slice position and volume ID arrays
                for i in range(B):
                    num_slices = seq_lengths[i].item()
                    slice_embs = embeddings[i, :num_slices, :]  # [num_slices, embed_dim]
                    all_embeddings.append(slice_embs.float())
                    
                    # Repeat labels for each slice
                    if len(batch_labels.shape) == 1:
                        slice_labels = batch_labels[i].unsqueeze(0).expand(num_slices)
                    else:
                        slice_labels = batch_labels[i].unsqueeze(0).expand(num_slices, -1)
                    all_labels.append(slice_labels)
                    
                    # Track slice positions (normalized to range 0-1)
                    positions = torch.arange(num_slices, device=accelerator.device).float() / max(num_slices - 1, 1)
                    all_slice_positions.append(positions)
                    
                    # Track volume IDs
                    volume_ids = torch.full((num_slices,), volume_counter + i, device=accelerator.device)
                    all_volume_ids.append(volume_ids)
                    
                    # Extend sample IDs for each slice
                    all_sample_ids.extend([batch_sample_ids[i]] * num_slices)
                
                volume_counter += B
            else:
                # Get aggregated embeddings [B, embed_dim]
                embeddings = cobra_model(features, seq_lengths=seq_lengths)
                all_embeddings.append(embeddings.float())
                all_labels.append(batch_labels)
                all_sample_ids.extend(batch_sample_ids)
    
    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    # Gather from all processes
    embeddings = accelerator.gather_for_metrics(embeddings)
    labels = accelerator.gather_for_metrics(labels)
    
    if slice_level:
        slice_positions = torch.cat(all_slice_positions, dim=0)
        volume_ids = torch.cat(all_volume_ids, dim=0)
        slice_positions = accelerator.gather_for_metrics(slice_positions)
        volume_ids = accelerator.gather_for_metrics(volume_ids)
        return (
            embeddings.cpu().numpy(), 
            labels.cpu().numpy(), 
            all_sample_ids,
            slice_positions.cpu().numpy(),
            volume_ids.cpu().numpy(),
        )
    
    return embeddings.cpu().numpy(), labels.cpu().numpy(), all_sample_ids, None, None


def main():
    parser = argparse.ArgumentParser(description="Extract COBRA embeddings and visualize with UMAP")
    
    parser.add_argument("--checkpoint-path", type=str, required=True, help="Path to COBRA checkpoint")
    parser.add_argument("--feat-dir", type=str, required=True, help="Directory with precomputed slice features")
    parser.add_argument("--annotations-path", type=str, required=True, help="Path to annotations CSV")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--plane", type=str, default="sagittal", help="View plane")
    parser.add_argument("--fm-model-names", type=str, default="dinov2", help="FM model names (space-separated)")
    parser.add_argument("--target-labels", type=str, nargs="+", required=True, help="Target label columns (e.g., ACL Meniscus)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fm-pooling", type=str, default="avg_pool", choices=["avg_pool", "attention"],
                        help="FM pooling: 'avg_pool' (average pooling) or 'attention' (requires fine-tuning).")
    parser.add_argument("--sequence-encoder", type=str, default="mamba2", choices=["mamba2", "transformer"])
    parser.add_argument("--slice-pooling", type=str, default="abmil", choices=["abmil", "cls"])
    parser.add_argument("--title", type=str, default="", help="Plot title")
    parser.add_argument("--save-embeddings", action="store_true", help="Save embeddings to npz file")
    
    # Slice-level analysis arguments
    parser.add_argument("--slice-level", action="store_true", 
                        help="Extract slice-level embeddings before aggregation")
    parser.add_argument("--color-by", type=str, default="label",
                        choices=["label", "slice_position", "volume"],
                        help="Color scheme for UMAP: 'label' (pathology), 'slice_position' (relative position), 'volume' (sample ID)")
    
    args = parser.parse_args()
    
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
        logger.info("COBRA Embedding Extraction & Clustering")
        logger.info("=" * 60)
        logger.info(f"Checkpoint: {args.checkpoint_path}")
        logger.info(f"Feature dir: {args.feat_dir}")
        logger.info(f"Annotations: {args.annotations_path}")
        logger.info(f"Target labels: {args.target_labels}")
        logger.info(f"Split: {args.split}, Plane: {args.plane}")
        logger.info(f"FM pooling: {args.fm_pooling}, Slice pooling: {args.slice_pooling}")
        logger.info(f"Slice-level: {args.slice_level}, Color by: {args.color_by}")
        logger.info("=" * 60)
    
    # Load COBRA
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
    
    # Determine task type based on number of target labels
    task = "multilabel" if len(args.target_labels) > 1 else "binary"
    
    # Create dataset with all target labels
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
    
    # Extract embeddings
    embeddings, labels, sample_ids, slice_positions, volume_ids = extract_embeddings(
        cobra_model, dataloader, accelerator, slice_level=args.slice_level
    )
    
    if not accelerator.is_main_process:
        return
    
    logger.info(f"Extracted embeddings: {embeddings.shape}")
    logger.info(f"Labels shape: {labels.shape}")
    if args.slice_level:
        logger.info(f"Slice positions shape: {slice_positions.shape}")
        logger.info(f"Volume IDs shape: {volume_ids.shape}")
    
    # Save embeddings if requested
    if args.save_embeddings:
        suffix = "_slice_level" if args.slice_level else ""
        npz_path = os.path.join(args.output_dir, f"embeddings{suffix}.npz")
        save_dict = {
            "embeddings": embeddings,
            "labels": labels,
            "sample_ids": np.array(sample_ids, dtype=object),
            "target_labels": np.array(args.target_labels, dtype=object),
        }
        if args.slice_level:
            save_dict["slice_positions"] = slice_positions
            save_dict["volume_ids"] = volume_ids
        np.savez(npz_path, **save_dict)
        logger.info(f"Saved embeddings to {npz_path}")
    
    # Determine coloring for UMAP
    if args.slice_level and args.color_by == "slice_position":
        # Color by relative slice position
        plot_embedding_clustering(
            embeddings=embeddings,
            output_dir=args.output_dir,
            labels=slice_positions,
            label_names=["Slice Position"],
            title=args.title or "Slice Embeddings (colored by position)",
            filename=f"umap_{args.plane}_{args.split}_slice_position.png",
            continuous_color=True,
        )
    elif args.slice_level and args.color_by == "volume":
        # Color by volume/sample ID
        plot_embedding_clustering(
            embeddings=embeddings,
            output_dir=args.output_dir,
            labels=volume_ids,
            label_names=["Volume ID"],
            title=args.title or "Slice Embeddings (colored by volume)",
            filename=f"umap_{args.plane}_{args.split}_volume.png",
            continuous_color=True,
        )
    else:
        # Color by pathology labels
        suffix = "_slice_level" if args.slice_level else ""
        plot_embedding_clustering(
            embeddings=embeddings,
            output_dir=args.output_dir,
            labels=labels,
            label_names=[label.capitalize() for label in args.target_labels],
            title=args.title,
            filename=f"umap_{args.plane}_{args.split}{suffix}.png",
        )

if __name__ == "__main__":
    main()
