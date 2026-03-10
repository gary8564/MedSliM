"""
COBRA Embedding Extraction and UMAP/t-SNE Clustering.

Extracts volume-level and slice-level embeddings using COBRA and visualizes clustering.
Always loads both train and test splits from the feature directory.

Usage:
    # Volume embeddings colored by pathology
    python -m med_slim.eval.embed_cluster \
        --experiment-dir /path/to/experiment_output
    # or 
    python -m med_slim.eval.embed_cluster \
        --checkpoint-path /path/to/checkpoint.pth.tar \
        --feat-dir /path/to/features \
        --annotations-dir /path/to/preprocessed/MRNet \
        --dataset-name MRNet \
        --output-dir ./outputs

    # Slice-level embeddings colored by pathology
    python -m med_slim.eval.embed_cluster \
        --experiment-dir /path/to/experiment_output --slice-level
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
from med_slim.eval.load_cobra import load_pretrained_cobra, load_cobra_from_experiment
from med_slim.utils.viz.cluster import plot_embedding_clustering
from med_slim.utils.label_metadata import (
    get_dataset_metadata,
    get_annotation_paths_by_split,
    build_binary_label_names,
    build_multilabel_display_names,
    build_multiclass_label_names,
)
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
        embeddings: [N, embed_dim] numpy array for volume-level, [total_slices, D] for slice-level
        labels: [N] or [N, num_labels] numpy array for volume, repeated per slice for slice-level
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
            batch_labels = batch["labels"]
            batch_sample_ids = batch["sample_ids"]
            
            if slice_level:
                embeddings = cobra_model(features, seq_lengths=seq_lengths, return_slice_embeddings=True)
                B = embeddings.shape[0]
                
                for i in range(B):
                    num_slices = seq_lengths[i].item()
                    all_embeddings.append(embeddings[i, :num_slices, :].float())
                    
                    if len(batch_labels.shape) == 1:
                        all_labels.append(batch_labels[i].unsqueeze(0).expand(num_slices))
                    else:
                        all_labels.append(batch_labels[i].unsqueeze(0).expand(num_slices, -1))
                    
                    all_sample_ids.extend([batch_sample_ids[i]] * num_slices)
            else:
                embeddings = cobra_model(features, seq_lengths=seq_lengths)
                all_embeddings.append(embeddings.float())
                all_labels.append(batch_labels)
                all_sample_ids.extend(batch_sample_ids)
    
    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    embeddings = accelerator.gather_for_metrics(embeddings)
    labels = accelerator.gather_for_metrics(labels)
    
    return embeddings.cpu().numpy(), labels.cpu().numpy(), all_sample_ids


def main():
    parser = argparse.ArgumentParser(description="Extract COBRA embeddings and visualize with UMAP")
    
    parser.add_argument("--experiment-dir", type=str, default=None,
                        help="Path to a linear-probing experiment directory. "
                             "Load model, features, annotations, and labels from config.yml and ckpt/classifier.pt directly.")
    # Optional args if experiment-dir is not provided
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Path to COBRA checkpoint")
    parser.add_argument("--feat-dir", type=str, default=None, help="Directory with precomputed slice features")
    parser.add_argument("--annotations-dir", type=str, default=None,
                        help="Directory containing split annotation CSVs "
                             "(e.g. train.csv/test.csv or train_multiclass.csv/test_multiclass.csv)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--dataset-name", type=str, default=None,
                        help="Dataset name (e.g. MRNet, SKM-TEA, kneeMRI)")
    parser.add_argument("--plane", type=str, default=None, help="View plane")
    parser.add_argument("--fm-model-names", type=str, default=None, help="FM model names (space-separated)")
    parser.add_argument("--target-labels", type=str, nargs="+", default=None,
                        help="Target label columns to override the ones specified in `eval_datasets.yaml`")
    parser.add_argument("--task", type=str, default=None, choices=["binary", "multiclass", "multilabel"],
                        help="Classification task type to override the one specified in `eval_datasets.yaml`")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fm-pooling", type=str, default=None, choices=["avg_pool", "attention"])
    parser.add_argument("--sequence-encoder", type=str, default=None, choices=["mamba2", "transformer"])
    parser.add_argument("--slice-pooling", type=str, default=None, choices=["abmil", "cls"])
    parser.add_argument("--save-embeddings", action="store_true", help="Save embeddings to npz file")
    parser.add_argument("--method", type=str, default="umap", choices=["umap", "tsne"],
                        help="Dimensionality reduction method (default: umap)")
    
    # UMAP hyperparameters
    parser.add_argument("--n-neighbors", type=int, default=15,
                        help="UMAP n_neighbors: larger values capture more global structure (default: 15)")
    parser.add_argument("--min-dist", type=float, default=0.1,
                        help="UMAP min_dist: lower values create tighter clusters (default: 0.1)")
    
    # t-SNE hyperparameters
    parser.add_argument("--perplexity", type=float, default=30,
                        help="t-SNE perplexity: related to nearest neighbors, try 20-50 (default: 30)")
    
    # Slice-level analysis arguments
    parser.add_argument("--slice-level", action="store_true", 
                        help="Extract slice-level embeddings before aggregation")
    parser.add_argument("--supervised", action="store_true",
                        help="Use supervised UMAP (labels guide the layout). "
                             "Only affects label-colored plots, not continuous-color plots.")
    
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
        output_dir = args.output_dir or os.path.join(args.experiment_dir, "embed_cluster")
        annotations_dir = args.annotations_dir or exp_cfg.get("annotations_dir")
        if not annotations_dir:
            raise ValueError(
                "Cannot determine annotations_dir from experiment config. "
                "Pass --annotations-dir explicitly."
            )
    else:
        if not args.checkpoint_path:
            parser.error("--checkpoint-path is required when --experiment-dir is not used")
        if not args.feat_dir:
            parser.error("--feat-dir is required when --experiment-dir is not used")
        if not args.annotations_dir:
            parser.error("--annotations-dir is required when --experiment-dir is not used")
        if not args.output_dir:
            parser.error("--output-dir is required when --experiment-dir is not used")

        feat_dir = args.feat_dir
        plane = args.plane or "sagittal"
        model_names = args.fm_model_names.split() if args.fm_model_names else ["dinov2"]
        dataset_name = args.dataset_name
        output_dir = args.output_dir
        annotations_dir = args.annotations_dir

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
            slice_pooling=args.slice_pooling or "abmil",
        )
        cobra_model = cobra_model.to(accelerator.device)
        cobra_model.eval()

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
    annotations_by_split = get_annotation_paths_by_split(
        annotations_dir, task, ["train", "test"]
    )
    splits = list(annotations_by_split.keys())

    if accelerator.is_main_process:
        logger.info("=" * 60)
        logger.info("COBRA Embedding Extraction & Clustering")
        logger.info("=" * 60)
        if args.experiment_dir:
            logger.info(f"Experiment dir: {args.experiment_dir}")
        logger.info(f"Feature dir: {feat_dir}")
        logger.info(f"Target labels: {target_labels}")
        logger.info(f"Plane: {plane}")
        logger.info(f"Task: {task}, Slice-level: {args.slice_level}, Method: {args.method}, Supervised: {args.supervised}")
        logger.info("=" * 60)

    cobra_model = accelerator.prepare(cobra_model)

    # Collect embeddings across splits
    vol_embeddings_list, vol_labels_list, vol_sample_ids = [], [], []
    slice_embeddings_list, slice_labels_list, slice_sample_ids = [], [], []

    for split in splits:
        dataset = FeatClassificationDataset(
            feat_dir=feat_dir,
            slice_encoder_models=model_names,
            view_plane=plane,
            split=split,
            annotations_path=str(annotations_by_split[split]),
            task=task,
            target_columns=target_labels,
        )
        if accelerator.is_main_process:
            logger.info(f"Loaded {len(dataset)} samples from split '{split}'")

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=linear_classifier_collate_fn,
            num_workers=4,
        )
        dataloader = accelerator.prepare(dataloader)

        # Volume-level embeddings
        vol_embs, vol_labs, vol_sids = extract_embeddings(
            cobra_model, dataloader, accelerator, slice_level=False
        )
        vol_embeddings_list.append(vol_embs)
        vol_labels_list.append(vol_labs)
        vol_sample_ids.extend(vol_sids)

        # Slice-level embeddings
        if args.slice_level:
            sl_embs, sl_labs, sl_sids = extract_embeddings(
                cobra_model, dataloader, accelerator, slice_level=True
            )
            slice_embeddings_list.append(sl_embs)
            slice_labels_list.append(sl_labs)
            slice_sample_ids.extend(sl_sids)

    if not accelerator.is_main_process:
        return

    vol_embeddings = np.concatenate(vol_embeddings_list, axis=0)
    vol_labels = np.concatenate(vol_labels_list, axis=0)
    logger.info(f"Total volume embeddings: {vol_embeddings.shape}")

    # Save embeddings if requested
    if args.save_embeddings:
        save_dict = {
            "embeddings": vol_embeddings,
            "labels": vol_labels,
            "sample_ids": np.array(vol_sample_ids, dtype=object),
            "target_labels": np.array(target_labels, dtype=object),
        }
        if args.slice_level:
            save_dict["slice_embeddings"] = np.concatenate(slice_embeddings_list, axis=0)
            save_dict["slice_labels"] = np.concatenate(slice_labels_list, axis=0)
        npz_path = os.path.join(output_dir, "embeddings.npz")
        np.savez(npz_path, **save_dict)
        logger.info(f"Saved embeddings to {npz_path}")

    # Build label names from dataset metadata (or fallback formatting)
    display_map = ds_meta.get("label_display_names") if ds_meta else None
    multiclass_maps = ds_meta.get("multiclass_label_maps") if ds_meta else None

    if task == "binary":
        vol_label_names = build_binary_label_names(target_labels[0], display_map)
    elif task == "multiclass":
        vol_label_names = build_multiclass_label_names(target_labels[0], multiclass_maps)
        if not vol_label_names:
            unique_vals = sorted(np.unique(vol_labels).astype(int))
            vol_label_names = [f"Class {v}" for v in range(max(unique_vals) + 1)]
            logger.warning(
                f"No multiclass display names for column '{target_labels[0]}'. "
                f"Using auto-generated names: {vol_label_names}. "
                f"Add entries to `eval_datasets.yaml` for readable legends."
            )
    else:
        vol_label_names = build_multilabel_display_names(target_labels, display_map)

    umap_kwargs = {"n_neighbors": args.n_neighbors, "min_dist": args.min_dist}
    tsne_kwargs = {"perplexity": args.perplexity}
    sup_tag = "_supervised" if args.supervised else ""

    # Plot 1: Volume embeddings colored by pathology
    plot_embedding_clustering(
        embeddings=vol_embeddings,
        output_dir=output_dir,
        labels=vol_labels,
        label_names=vol_label_names,
        title=f"{dataset_name}",
        filename=f"{dataset_name}_{args.method}_{plane}_embedding{sup_tag}.png",
        method=args.method,
        umap_kwargs=umap_kwargs,
        tsne_kwargs=tsne_kwargs,
        supervised=args.supervised,
    )

    # Slice-level plot: colored by pathology
    if args.slice_level:
        slice_embeddings = np.concatenate(slice_embeddings_list, axis=0)
        slice_labels = np.concatenate(slice_labels_list, axis=0)
        logger.info(f"Total slice embeddings: {slice_embeddings.shape}")

        plot_embedding_clustering(
            embeddings=slice_embeddings,
            output_dir=output_dir,
            labels=slice_labels,
            label_names=vol_label_names,
            title=f"{dataset_name}",
            filename=f"{dataset_name}_{args.method}_{plane}_slice_embedding{sup_tag}.png",
            method=args.method,
            umap_kwargs=umap_kwargs,
            tsne_kwargs=tsne_kwargs,
            supervised=args.supervised,
        )

if __name__ == "__main__":
    main()
