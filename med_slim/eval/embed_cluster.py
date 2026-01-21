"""
COBRA Embedding Extraction and UMAP Clustering.

Extracts stack-level embeddings using COBRA and visualizes clustering by pathology labels.

Usage:
    python -m med_slim.eval.embed_cluster \
        --checkpoint-path /path/to/cobra/checkpoint.pth.tar \
        --feat-dir /path/to/precomputed/features \
        --annotations-path /path/to/annotations.csv \
        --output-dir ./outputs \
        --target-labels acl meniscus \
        --plane sagittal \
        --title "MRNet Embedding Space"
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
):
    """
    Extract COBRA embeddings from a dataloader.
    
    Returns:
        embeddings: [N, embed_dim] numpy array
        labels: [N] or [N, num_labels] numpy array
        sample_ids: List of sample IDs
    """
    cobra_model.eval()
    all_embeddings = []
    all_labels = []
    all_sample_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings", disable=not accelerator.is_main_process):
            seq_lengths = batch["seq_lengths"].to(accelerator.device)
            features = [f.to(accelerator.device, dtype=next(cobra_model.parameters()).dtype) for f in batch["features"]]
            
            embeddings = cobra_model(features, seq_lengths=seq_lengths)
            
            all_embeddings.append(embeddings.float())
            all_labels.append(batch["labels"])
            all_sample_ids.extend(batch["sample_ids"])
    
    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    # Gather from all processes
    embeddings = accelerator.gather_for_metrics(embeddings)
    labels = accelerator.gather_for_metrics(labels)
    
    return embeddings.cpu().numpy(), labels.cpu().numpy(), all_sample_ids


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
    parser.add_argument("--fm-pooling", type=str, default="mean", choices=["mean", "concat"],
                        help="FM pooling: 'mean' or 'concat'. Note: 'attention' not supported for frozen COBRA.")
    parser.add_argument("--sequence-encoder", type=str, default="mamba2", choices=["mamba2", "transformer"])
    parser.add_argument("--slice-pooling", type=str, default="abmil", choices=["abmil", "cls"])
    parser.add_argument("--title", type=str, default="", help="Plot title")
    parser.add_argument("--save-embeddings", action="store_true", help="Save embeddings to npz file")
    
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
    embeddings, labels, sample_ids = extract_embeddings(cobra_model, dataloader, accelerator)
    
    if not accelerator.is_main_process:
        return
    
    logger.info(f"Extracted embeddings: {embeddings.shape}")
    logger.info(f"Labels shape: {labels.shape}")
    
    # Save embeddings if requested
    if args.save_embeddings:
        npz_path = os.path.join(args.output_dir, "embeddings.npz")
        np.savez(
            npz_path,
            embeddings=embeddings,
            labels=labels,
            sample_ids=np.array(sample_ids, dtype=object),
            target_labels=np.array(args.target_labels, dtype=object),
        )
        logger.info(f"Saved embeddings to {npz_path}")
    
    # Plot UMAP
    plot_embedding_clustering(
        embeddings=embeddings,
        output_dir=args.output_dir,
        labels=labels,
        label_names=[label.capitalize() for label in args.target_labels],
        title=args.title,
        filename=f"umap_{args.plane}_{args.split}.png",
    )

if __name__ == "__main__":
    main()
