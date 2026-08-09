"""
KNN Evaluation for pretrained MedSliM SSL Model.

DINOv2-style temperature-weighted KNN classification on COBRA embeddings.

References:
    Adapted from https://github.com/facebookresearch/dinov2/blob/main/dinov2/eval/knn.py
    and https://github.com/qj474765/rad_dino/blob/main/rad_dino/eval/feature_extractor.py
"""

import os
import argparse
import yaml
import json
import logging
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from accelerate import Accelerator
from typing import Dict, List, Optional, Tuple

from med_slim.model.sequence_encoder.cobra import Cobra, _resolve_pooling_target
from med_slim.data.feat_dataset import (
    FeatClassificationDataset,
    linear_classifier_collate_fn,
)
from med_slim.eval.load_cobra import load_pretrained_cobra, resolve_eval_fm_ids, resolve_raw_aggregation_fm
from med_slim.utils.metrics.linear import get_num_classes
from med_slim.utils.viz.linear import compute_and_visualize_metrics
from med_slim.utils.label_metadata import (
    get_dataset_metadata,
    get_annotation_paths_by_split,
    build_multiclass_label_names,
    build_run_label_tag,
)
from med_slim.logging.setup import init_logging

init_logging()
logger = logging.getLogger(__name__)

CURR_TIME = datetime.now().strftime("%Y-%m-%d-%H:%M")


# COBRA Feature Extraction
@torch.no_grad()
def extract_cobra_features(
    cobra_model: Cobra,
    dataloader: DataLoader,
    accelerator: Accelerator,
    normalize: bool = True,
    fm_ids: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Extract volume-level embeddings from COBRA and optionally L2-normalize.

    Args:
        cobra_model: Pretrained COBRA model in inference mode.
        dataloader: DataLoader yielding batches from FeatClassificationDataset.
        accelerator: HuggingFace Accelerator.
        normalize: If True, L2-normalize feature vectors.

    Returns:
        features: [N, output_dim] numpy array.
        labels: [N, ...] numpy array.
        sample_ids: List of sample IDs.
    """
    cobra_model.eval()
    all_features = []
    all_labels = []
    all_sample_ids = []

    model_dtype = next(cobra_model.parameters()).dtype
    fm_ids_tensor = None if fm_ids is None else torch.as_tensor(fm_ids, dtype=torch.long, device=accelerator.device)

    for batch in tqdm(dataloader, desc="Extracting COBRA features",
                      disable=not accelerator.is_main_process):
        seq_lengths = batch["seq_lengths"].to(accelerator.device)
        physical_positions = batch.get("physical_positions")
        if physical_positions is not None:
            physical_positions = physical_positions.to(accelerator.device, dtype=torch.float32)
        features = [f.to(accelerator.device, dtype=model_dtype) for f in batch["features"]]

        embeddings = cobra_model(
            features,
            seq_lengths=seq_lengths,
            physical_positions=physical_positions,
            fm_ids=fm_ids_tensor,
        )  # [B, output_dim]
        embeddings = embeddings.float()

        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        all_features.append(embeddings)
        all_labels.append(batch["labels"])
        all_sample_ids.extend(batch["sample_ids"])

    features = accelerator.gather_for_metrics(torch.cat(all_features, dim=0))
    labels = accelerator.gather_for_metrics(torch.cat(all_labels, dim=0))

    return features.cpu().numpy(), labels.cpu().numpy(), all_sample_ids


# KNN Classification
def knn_classify(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    nb_knn: int,
    temperature: float,
    num_classes: int,
) -> torch.Tensor:
    """
    DINOv2-style temperature-weighted KNN classification.

    1. L2-normalise features.
    2. Compute cosine similarity.
    3. Select top-K neighbours.
    4. Apply temperature-scaled softmax over similarities.
    5. Weighted one-hot voting to obtain class probabilities.

    Args:
        train_features: [N_train, D] training features.
        train_labels: [N_train] integer class labels.
        test_features: [N_test, D] test features.
        nb_knn: Number of nearest neighbours (K).
        temperature: Softmax temperature (DINOv2 default: 0.07).
        num_classes: Total number of classes.

    Returns:
        Class probability tensor with shape [N_test, num_classes].
    """
    train_features = F.normalize(train_features, dim=1)
    test_features = F.normalize(test_features, dim=1)

    similarities = test_features @ train_features.T  # [N_test, N_train]

    topk_sims, topk_indices = similarities.topk(nb_knn, dim=1)  # [N_test, K]
    topk_labels = train_labels[topk_indices]  # [N_test, K]

    weights = F.softmax(topk_sims / temperature, dim=1)  # [N_test, K]
    one_hot = F.one_hot(topk_labels, num_classes).float()  # [N_test, K, num_classes]
    probas = (one_hot * weights.unsqueeze(-1)).sum(dim=1)  # [N_test, num_classes]

    return probas


# KNN Evaluation Pipeline
def run_knn_evaluation(
    cobra_model: Cobra,
    train_dataset: FeatClassificationDataset,
    test_dataset: FeatClassificationDataset,
    cfg: Dict,
    accelerator: Accelerator,
    output_dir: str,
    nb_knn_list: List[int],
    temperature: float,
    fm_ids: Optional[List[int]] = None,
) -> Dict:
    """
    Run KNN evaluation pipeline.

    1. Extract COBRA embeddings for train and test sets.
    2. For each K value, run temperature-weighted KNN.
    3. Compute and save metrics.

    Returns:
        Dict with results per K value.
    """
    task = cfg["task"]
    target_labels = cfg["target_labels"]
    hyperparams = cfg.get("hyperparams", {})
    batch_size = hyperparams.get("batch_size", 32)

    if accelerator.is_main_process:
        logger.info(f"\n{'='*50}")
        logger.info("KNN Evaluation")
        logger.info(f"K values: {nb_knn_list}, Temperature: {temperature}")
        logger.info(f"{'='*50}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=linear_classifier_collate_fn,
        num_workers=hyperparams.get("num_workers", 4),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=linear_classifier_collate_fn,
        num_workers=hyperparams.get("num_workers", 4),
    )

    train_loader, test_loader = accelerator.prepare(train_loader, test_loader)

    # Extract features
    if accelerator.is_main_process:
        logger.info("Extracting training features...")
    train_features, train_labels, _ = extract_cobra_features(
        cobra_model, train_loader, accelerator, fm_ids=fm_ids
    )
    if accelerator.is_main_process:
        logger.info(f"Train features: {train_features.shape}")

    if accelerator.is_main_process:
        logger.info("Extracting test features...")
    test_features, test_labels, test_sample_ids = extract_cobra_features(
        cobra_model, test_loader, accelerator, fm_ids=fm_ids
    )
    if accelerator.is_main_process:
        logger.info(f"Test features: {test_features.shape}")

    if not accelerator.is_main_process:
        return {}

    # Determine number of classes
    if task == "binary":
        num_classes = 2
    else:
        num_classes = get_num_classes(task, target_labels, train_dataset)

    device = accelerator.device
    train_feat_t = torch.from_numpy(train_features).to(device)
    test_feat_t = torch.from_numpy(test_features).to(device)

    if task == "binary":
        train_labels_t = torch.from_numpy(
            train_labels.squeeze().astype(np.int64)
        ).to(device)
    else:
        train_labels_t = torch.from_numpy(
            train_labels.astype(np.int64)
        ).to(device)

    all_results = {}

    for k in nb_knn_list:
        logger.info(f"\nRunning KNN with K={k}, T={temperature}")

        probas = knn_classify(
            train_feat_t, train_labels_t, test_feat_t,
            k, temperature, num_classes,
        )
        probas_np = probas.cpu().numpy()

        # For binary, extract positive class probability
        if task == "binary":
            eval_probs = probas_np[:, 1]
        else:
            eval_probs = probas_np

        # Compute and visualize metrics
        k_output_dir = os.path.join(output_dir, f"knn_k{k}")
        os.makedirs(k_output_dir, exist_ok=True)

        eval_results = _compute_knn_predictions_and_metrics(
            all_probs=eval_probs,
            all_true=test_labels,
            sample_ids=test_sample_ids,
            task=task,
            target_labels=target_labels,
            output_dir=k_output_dir,
            class_names=cfg.get("class_names"),
        )

        # Save predictions and metrics
        table_dir = os.path.join(k_output_dir, "table")
        os.makedirs(table_dir, exist_ok=True)

        eval_results["predictions"].to_csv(
            os.path.join(table_dir, "predictions.csv"), index=False
        )

        if eval_results["metrics"]:
            save_metrics = _prepare_saving_metrics(eval_results["metrics"], task)
            save_metrics["k"] = k
            save_metrics["temperature"] = temperature

            with open(os.path.join(table_dir, "metrics.json"), "w") as f:
                json.dump(save_metrics, f, indent=4)

            auroc = save_metrics["AUROC"] if task == "binary" else save_metrics["overall"]["AUROC"]
            auprc = save_metrics["AUPRC"] if task == "binary" else save_metrics["overall"]["AUPRC"]

            logger.info(f"K={k}: AUROC={auroc:.4f}, AUPRC={auprc:.4f}")

            wandb.log({
                f"knn_k{k}/auroc": auroc,
                f"knn_k{k}/auprc": auprc,
            })

            all_results[f"k{k}"] = {
                "auroc": auroc,
                "auprc": auprc,
                "metrics": save_metrics,
            }

    # Log best K
    if all_results:
        best_k = max(all_results, key=lambda x: all_results[x]["auroc"])
        logger.info(f"\nBest K: {best_k} (AUROC={all_results[best_k]['auroc']:.4f})")
        wandb.run.summary["best_knn_k"] = best_k
        wandb.run.summary["best_knn_auroc"] = all_results[best_k]["auroc"]

    return all_results


def _compute_knn_predictions_and_metrics(
    all_probs: np.ndarray,
    all_true: np.ndarray,
    sample_ids: List,
    task: str,
    target_labels: List[str],
    output_dir: Optional[str] = None,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """Convert KNN probabilities to predictions and compute metrics."""
    import pandas as pd

    if task == "binary":
        all_probs = all_probs.squeeze(-1) if all_probs.ndim > 1 else all_probs
        all_preds = (all_probs >= 0.5).astype(int)
    elif task == "multilabel":
        all_preds = (all_probs >= 0.5).astype(int)
    else:
        all_preds = all_probs.argmax(axis=-1)

    viz_metrics = None
    if output_dir is not None:
        fig_dir = os.path.join(output_dir, "fig")

        if task == "binary":
            viz_labels = target_labels[:1]
        elif task == "multiclass":
            viz_labels = class_names if class_names else [f"class_{i}" for i in range(all_probs.shape[-1])]
        else:
            viz_labels = target_labels

        viz_metrics = compute_and_visualize_metrics(
            y_true=all_true,
            y_pred_prob=all_probs,
            task=task,
            class_labels=viz_labels,
            output_dir=fig_dir,
        )

    if task == "binary":
        df_results = pd.DataFrame({
            "exam_id": sample_ids,
            "true_labels": all_true.tolist(),
            "pred_labels": all_preds.tolist(),
            "pred_probs": all_probs.tolist(),
        })
    elif task == "multilabel":
        df_results = pd.DataFrame({"exam_id": sample_ids})
        df_results["true_labels"] = [list(map(int, row)) for row in all_true]
        df_results["pred_labels"] = [
            [target_labels[i] for i, p in enumerate(row) if p >= 0.5]
            for row in all_probs
        ]
        df_results["pred_probs"] = [list(row) for row in all_probs]
    else:
        df_results = pd.DataFrame({
            "exam_id": sample_ids,
            "true_labels": all_true.tolist(),
            "pred_labels": all_preds.tolist(),
            "pred_probs": [list(row) for row in all_probs],
        })

    return {"predictions": df_results, "metrics": viz_metrics}


def _prepare_saving_metrics(metrics: Dict, task: str) -> Dict:
    """Prepare metrics dictionary for JSON serialization."""
    save_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, (np.floating, np.integer)):
            save_metrics[key] = float(value)
        elif isinstance(value, np.ndarray):
            save_metrics[key] = value.tolist()
        elif isinstance(value, dict):
            save_metrics[key] = _prepare_saving_metrics(value, task)
        else:
            save_metrics[key] = value
    return save_metrics


# Main Entry Point
def main(args):
    """Main function for KNN evaluation."""
    accelerator = Accelerator()

    # Load config (reuses the same linear_classifier config format)
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    dataset_name = cfg.get("feat_dataset", {}).get("dataset_name")
    if dataset_name is None:
        raise ValueError("Missing required field `dataset_name` in config.")
    ds_meta = get_dataset_metadata(dataset_name)
    cfg["task"] = ds_meta["task"]
    cfg["target_labels"] = ds_meta["target_labels"]
    if cfg["task"] == "multiclass":
        if "multiclass_label_maps" not in ds_meta:
            raise ValueError("Missing `multiclass_label_maps` in eval_datasets.yaml.")
        cfg["class_names"] = build_multiclass_label_names(
            cfg["target_labels"][0], ds_meta["multiclass_label_maps"]
        )

    annotations_dir = cfg.get("annotations_dir")
    if not annotations_dir:
        raise ValueError("Missing required field `annotations_dir` in config.")
    annot_files = get_annotation_paths_by_split(
        annotations_dir, cfg["task"], splits=["train", "test"],
    )
    cfg["train_annots"] = str(annot_files["train"])
    cfg["test_annots"] = str(annot_files["test"])

    view_planes = cfg["feat_dataset"]["plane"]
    if args.fm_model_names:
        model_names = args.fm_model_names.split()
    else:
        model_names = cfg["feat_dataset"]["model_name"]
    if isinstance(model_names, str):
        model_names = [model_names]
        
    # Only single-view KNN is supported
    if len(view_planes) > 1:
        raise ValueError("Only single-view KNN is supported.")
    view_plane = view_planes[0]

    run_label_tag = build_run_label_tag(cfg["task"], cfg["target_labels"], view_plane)
    k_str = "_".join(str(k) for k in args.nb_knn)
    output_dir = os.path.join(
        cfg["output_dir"],
        f"knn_{run_label_tag}_k{k_str}_{CURR_TIME}",
    )

    checkpoint_path = args.checkpoint_path if args.checkpoint_path else cfg["checkpoint_path"]
    pretrain_config_path = Path(checkpoint_path).parent / "config.yaml"
    with open(pretrain_config_path, "r") as f:
        pretrain_cfg = yaml.safe_load(f)
    cobra_cfg = pretrain_cfg["model"]["cobra"]
    pretrain_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_slice_pooling = pretrain_state.get("pooling")
    cobra_cfg["regional_tokens"] = int(
        pretrain_state.get("regional_tokens", cobra_cfg.get("regional_tokens", 0))
    )
    cfg["cobra_config"] = cobra_cfg
    fm_pooling = (
        args.fm_pooling
        or pretrain_state.get("fm_pooling")
        or cobra_cfg.get("fm_pooling", "avg_pool")
    )
    eval_fm_ids = resolve_eval_fm_ids(
        fm_pooling, model_names, pretrain_cfg, pretrain_state
    )
    fm_id_order = pretrain_state.get("fm_id_order") or pretrain_cfg.get("feat_dataset", {}).get("model_name")

    resolved_pooling_target = _resolve_pooling_target(
        mode="inference",
        pooling_target=args.pooling_target,
        regional_tokens=cobra_cfg.get("regional_tokens", 0),
        slice_pooling=args.slice_pooling or checkpoint_slice_pooling or cobra_cfg.get("pooling", "abmil"),
    )

    # Determine raw FM output dimension and aggregation FM when pooling_target='raw'
    raw_aggregation_index, raw_output_dim, raw_aggregation_fm = resolve_raw_aggregation_fm(
        resolved_pooling_target, model_names, pretrain_cfg, args.raw_aggregation_fm
    )
    if accelerator.is_main_process and raw_aggregation_fm is not None:
        logger.info(
            f"pooling_target='raw': aggregating raw features from FM '{raw_aggregation_fm}' "
            f"(index {raw_aggregation_index} of {model_names})"
        )
    cfg["pooling_target"] = resolved_pooling_target
    cfg["raw_output_dim"] = raw_output_dim
    cfg["raw_aggregation_index"] = raw_aggregation_index
    cfg["raw_aggregation_fm"] = raw_aggregation_fm
    cfg["fm_pooling"] = fm_pooling
    cfg["fm_id_order"] = fm_id_order
    cfg["eval_fm_ids"] = eval_fm_ids

    # Initialize wandb
    if accelerator.is_main_process:
        wandb.init(
            project="medslim-knn",
            name=f"{dataset_name}_{run_label_tag}_knn",
            config={**cfg, "nb_knn": args.nb_knn, "temperature": args.temperature},
            dir=output_dir,
        )

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "config.yml"), "w") as f:
            yaml.dump(cfg, f)

    # Load pretrained COBRA model
    if accelerator.is_main_process:
        logger.info("Loading pretrained COBRA model...")
        logger.info(f"Checkpoint: {checkpoint_path}")
    cobra_model = load_pretrained_cobra(
        checkpoint_path=checkpoint_path,
        accelerator=accelerator,
        model_config=cobra_cfg,
        encoder_type=cfg.get("encoder_type", "momentum"),
        fm_pooling=fm_pooling,
        sequence_encoder=args.sequence_encoder,
        slice_pooling=args.slice_pooling,
        pooling_target=resolved_pooling_target,
        raw_output_dim=raw_output_dim,
        raw_aggregation_index=raw_aggregation_index,
    )
    cobra_model = cobra_model.to(accelerator.device)
    cobra_model.eval()
    cobra_model = accelerator.prepare(cobra_model)

    # Create datasets
    train_dataset = FeatClassificationDataset(
        feat_dir=cfg["feat_dataset"]["feat_dir"],
        slice_encoder_models=model_names,
        view_plane=view_plane,
        split="train",
        annotations_path=cfg["train_annots"],
        task=cfg["task"],
        target_columns=cfg["target_labels"],
        cache_in_memory=True,
    )
    test_dataset = FeatClassificationDataset(
        feat_dir=cfg["feat_dataset"]["feat_dir"],
        slice_encoder_models=model_names,
        view_plane=view_plane,
        split="test",
        annotations_path=cfg["test_annots"],
        task=cfg["task"],
        target_columns=cfg["target_labels"],
        cache_in_memory=True,
    )

    if accelerator.is_main_process:
        logger.info(f"View plane: {view_plane}")
        logger.info(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    run_knn_evaluation(
        cobra_model=cobra_model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        cfg=cfg,
        accelerator=accelerator,
        output_dir=output_dir,
        nb_knn_list=args.nb_knn,
        temperature=args.temperature,
        fm_ids=eval_fm_ids,
    )

    if accelerator.is_main_process:
        logger.info(f"\n{'='*50}")
        logger.info("KNN Evaluation Complete!")
        logger.info(f"Results saved to: {output_dir}")
        logger.info(f"{'='*50}")
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KNN Evaluation for MedSliM")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to evaluation config file (same format as linear_classifier.yml)",
    )
    parser.add_argument(
        "--checkpoint-path", type=str, default=None,
        help="Path to pretrained COBRA checkpoint. Overrides config if provided.",
    )
    parser.add_argument(
        "--fm-model-names", type=str, default=None,
        help="FM model names (space-separated). If not provided, uses config.",
    )
    parser.add_argument(
        "--nb-knn", nargs="+", type=int, default=[5, 10, 20, 50],
        help="Number of nearest neighbours to evaluate (default: 5 10 20 50).",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.07,
        help="Temperature for softmax voting (default: 0.07).",
    )
    parser.add_argument(
        "--sequence-encoder", type=str, choices=["mamba2", "transformer"], default=None,
    )
    parser.add_argument(
        "--slice-pooling", type=str, choices=["abmil", "cls"], default=None,
    )
    parser.add_argument(
        "--pooling-target", type=str, choices=["post_encoder", "post_embed", "raw"], default=None,
        help="Which representation level ABMIL attention weights aggregate. "
             "Raw pooling uses original FM embeddings and flattens tiled tokens when present. "
             "With multiple eval FMs, shared multi-FM attention is transferred to the FM explicitly specified by --raw-aggregation-fm. "
             "If using default None, Cobra resolves to raw for global-only FM caches and post_embed for tiled multi-crop CLS caches.",
    )
    parser.add_argument(
        "--raw-aggregation-fm", type=str, default=None,
        help="Name of the evaluated FM whose raw features are aggregated when --pooling-target=raw. "
             "Required when more than one FM is evaluated (i.e., --fm-model-names has more than one entry).",
    )
    parser.add_argument(
        "--fm-pooling", type=str, choices=["avg_pool", "router"], default=None,
        help="FM fusion mode. If omitted, uses the checkpoint's saved fm_pooling when available.",
    )
    args = parser.parse_args()
    main(args)
