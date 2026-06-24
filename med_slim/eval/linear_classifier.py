"""
Linear Probing Evaluation for pretrained MedSliM SSL Model.

- Single-channel classification: Train linear classifier head one one specific (sequence, plane).
- Multi-channel classification (logistic regression ensemble):
  1. Train independent single-channel classifiers for each channel
  2. Collect predictions from each channel
  3. Fit logistic regression on stacked predictions
  4. Evaluate ensemble on test set

A "channel" denotes a possible (sequence, plane) combination.  
For single-sequence datasets, a channel is simply a view plane (e.g. sagittal, coronal, axial).
For multi-sequence datasets, each sequence x plane pair is an independent channel (e.g. "DESS_E1/sagittal", "DESS_E2/sagittal").
"""
import os
import argparse
import yaml
import json
import logging
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from pathlib import Path
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from datetime import datetime
from accelerate import Accelerator
from typing import Dict, List, Optional, Tuple
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from transformers import get_cosine_schedule_with_warmup

from med_slim.model.sequence_encoder.cobra import Cobra, _resolve_pooling_target
from med_slim.model.attention_pooling.abmil import BatchedABMIL
from med_slim.data.feat_dataset import (
    FeatClassificationDataset, 
    MultiViewFeatClassificationDataset,
    linear_classifier_collate_fn,
    multiview_classifier_collate_fn,
)
from med_slim.eval.load_cobra import load_pretrained_cobra, resolve_eval_fm_ids
from med_slim.utils.callbacks.early_stopping import EarlyStopping
from med_slim.utils.metrics.linear import get_loss_criterion, get_eval_metrics, get_num_classes, compute_class_weights_for_weighted_loss
from med_slim.utils.viz.linear import compute_and_visualize_metrics, compute_youden_thresholds
from med_slim.utils.label_metadata import get_dataset_metadata, get_annotation_paths_by_split, build_multiclass_label_names
from codecarbon import EmissionsTracker
from med_slim.logging.setup import init_logging

init_logging()
logger = logging.getLogger(__name__)

CURR_TIME = datetime.now().strftime("%Y-%m-%d-%H:%M")
JOB_ID = os.environ.get("SLURM_JOB_ID", str(os.getpid()))


# =============================================================================
# Few-shot subsampling, seeding, and subset-aware class weights
# =============================================================================
def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for a single repeat."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _labels_for_indices(dataset, indices, task: str) -> np.ndarray:
    sample_ids = dataset.sample_ids
    return np.array([
        dataset._get_label(sample_ids[int(i)]).numpy() for i in indices
    ])


def _label_positive_counts(labels: np.ndarray, task: str, target_labels: List[str]) -> Dict[str, int]:
    """Per-label positive counts for binary/multilabel, or per-class counts for multiclass."""
    counts: Dict[str, int] = {}
    if task == "multiclass":
        classes, class_counts = np.unique(labels, return_counts=True)
        for cls, cnt in zip(classes.tolist(), class_counts.tolist()):
            counts[f"class_{int(cls)}"] = int(cnt)
    else:
        label_matrix = labels if task == "multilabel" else labels.reshape(-1, 1)
        for col in range(label_matrix.shape[1]):
            name = target_labels[col] if col < len(target_labels) else f"label_{col}"
            counts[name] = int((label_matrix[:, col] == 1).sum())
    return counts


def subsample_train_indices(
    dataset,
    candidate_idx,
    task: str,
    train_fraction: float,
    seed: int,
) -> np.ndarray:
    """
    Stratified subsample of candidate training indices down to `train_fraction`.

    Only training indices should be passed here; validation/test indices must
    never be subsampled. Falls back to a deterministic random selection when
    stratification is infeasible for the requested fraction (e.g. very small
    fractions with sparse positives).
    """
    candidate_idx = np.asarray(candidate_idx)
    if not 0.0 < train_fraction <= 1.0:
        raise ValueError(f"train_fraction must be in (0, 1], got {train_fraction}")

    n_total = len(candidate_idx)
    n_select = max(1, int(round(train_fraction * n_total)))
    if n_select >= n_total:
        return candidate_idx

    labels = _labels_for_indices(dataset, candidate_idx, task)
    total_indices = np.arange(n_total)

    try:
        if task == "multilabel":
            n_splits = max(2, min(n_total, int(round(1.0 / train_fraction))))
            mskf = MultilabelStratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=seed
            )
            # The held-out fold is ~1/n_splits of the data, i.e. ~train_fraction.
            _, train_frac_indices = next(iter(mskf.split(total_indices, labels)))
        else:
            train_frac_indices, _ = train_test_split(
                total_indices,
                train_size=n_select,
                stratify=labels,
                random_state=seed,
                shuffle=True,
            )
    except ValueError as exc:
        logger.warning(
            f"Stratified subsampling failed ({exc}); falling back to random "
            f"selection for train_fraction={train_fraction}."
        )
        rng = np.random.default_rng(seed)
        train_frac_indices = rng.choice(total_indices, size=n_select, replace=False)

    return candidate_idx[np.sort(train_frac_indices)]


def compute_class_weights_from_indices(
    dataset,
    indices,
    task: str,
    target_labels: List[str],
    device: torch.device,
) -> torch.Tensor:
    """Compute weighted-loss class weights from the effective training subset."""
    labels = _labels_for_indices(dataset, indices, task)
    if task == "multiclass":
        classes, class_counts = np.unique(labels, return_counts=True)
        total = float(labels.shape[0])
        order = np.argsort(classes)
        weights = [total / float(class_counts[i]) for i in order]
        w = torch.tensor(weights, dtype=torch.float32)
    else:
        label_matrix = labels if task == "multilabel" else labels.reshape(-1, 1)
        weights = []
        for col in range(label_matrix.shape[1]):
            col_vals = label_matrix[:, col]
            pos = float((col_vals == 1).sum())
            neg = float((col_vals == 0).sum())
            if pos == 0:
                name = target_labels[col] if col < len(target_labels) else f"label_{col}"
                logger.warning(
                    f"Label '{name}' has 0 positives in the selected training subset; "
                    f"using pos_weight=1.0."
                )
                weights.append(1.0)
            else:
                weights.append(neg / pos)
        w = torch.tensor(weights, dtype=torch.float32)
    return w.squeeze().to(device)


def prepare_training_indices(
    dataset,
    candidate_idx,
    cfg: Dict,
    accelerator: Accelerator,
    train_fraction: float,
    subsample_seed: int,
    context: str = "",
) -> Tuple[np.ndarray, Optional[torch.Tensor]]:
    """Subsample candidate training indices and derive subset-aware class weights."""
    selected_idx = subsample_train_indices(
        dataset, candidate_idx, cfg["task"], train_fraction, subsample_seed
    )

    class_weights = None
    if cfg.get("weighted_loss"):
        class_weights = compute_class_weights_from_indices(
            dataset, selected_idx, cfg["task"], cfg["target_labels"], accelerator.device
        )

    if accelerator.is_main_process:
        labels = _labels_for_indices(dataset, selected_idx, cfg["task"])
        pos_counts = _label_positive_counts(labels, cfg["task"], cfg["target_labels"])
        ctx = f" [{context}]" if context else ""
        logger.info(
            f"Effective training samples{ctx}: {len(selected_idx)} (train_fraction={train_fraction}, positive counts={pos_counts})"
        )

    return selected_idx, class_weights


def aggregate_repeat_metrics(repeat_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate per-repeat test metrics into mean/std summaries."""
    aurocs = [m["test_auroc"] for m in repeat_metrics if m.get("test_auroc") is not None]
    auprcs = [m["test_auprc"] for m in repeat_metrics if m.get("test_auprc") is not None]
    summary: Dict[str, float] = {}
    if aurocs:
        summary["mean_auroc"] = float(np.mean(aurocs))
        summary["std_auroc"] = float(np.std(aurocs))
    if auprcs:
        summary["mean_auprc"] = float(np.mean(auprcs))
        summary["std_auprc"] = float(np.std(auprcs))
    return summary


# =============================================================================
# Classifier Head and Attention Aggregator
# =============================================================================
class ClassifierHead(nn.Module):
    """MLP classifier head for linear probing."""
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 512, dropout: float = 0.5):
        super().__init__()
        self.num_classes = num_classes
        output_dim = 1 if num_classes == 2 else num_classes
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class MultiViewAttentionAggregator(nn.Module):
    """
    Attention-based aggregator for multi-view embeddings.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.abmil = BatchedABMIL(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            n_heads=1,
            activation='softmax',
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Stacked view embeddings [B, num_views, embed_dim]
        
        Returns:
            aggregated: Attention-weighted embedding [B, embed_dim]
            attention_weights: Attention weights per view [B, num_views]
        """
        # Get attention weights from ABMIL: [B, num_views, 1]
        attention_weights = self.abmil(x)
        
        # Weighted aggregation via batch matrix multiplication
        # [B, 1, num_views] @ [B, num_views, embed_dim] -> [B, 1, embed_dim]
        aggregated = torch.bmm(attention_weights.transpose(2, 1), x).squeeze(1)
        
        return aggregated, attention_weights.squeeze(-1)


class SingleViewClassifier(nn.Module):
    """
    Single-view classifier that integrates COBRA encoder.
    
    Architecture:
        slice-level features (list of K tensors) -> COBRA -> volume-level embedding -> classifier -> logits
    """
    def __init__(
        self,
        cobra_model: Cobra,
        input_dim: int,
        num_classes: int,
        classifier_hidden_dim: int = 512,
        classifier_dropout: float = 0.5,
        freeze_cobra: bool = True,
        fm_ids: Optional[List[int]] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.freeze_cobra = freeze_cobra
        fm_id_tensor = None if fm_ids is None else torch.as_tensor(fm_ids, dtype=torch.long)
        self.register_buffer("fm_ids", fm_id_tensor, persistent=False)
        
        # COBRA encoder
        self.cobra = cobra_model
        if freeze_cobra:
            for param in self.cobra.parameters():
                param.requires_grad = False
        
        # Classifier head
        self.classifier = ClassifierHead(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=classifier_hidden_dim,
            dropout=classifier_dropout,
        )
    
    def forward(
        self,
        features: List[torch.Tensor],
        seq_lengths: torch.Tensor,
        physical_positions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: List of K tensors. 
                      Global-only caches have shape [B, num_slices, input_embed_dim]. 
                      Tiled multi-crop CLS caches have shape [B, num_slices, num_tiled_regions, input_embed_dim].
            seq_lengths: Sequence lengths [B]
            physical_positions: Physical slice positions [B, max_seq_len], used when Cobra has physical PE enabled.
        
        Returns:
            Dict with logits and embedding
        """
        # Keep COBRA in eval mode when frozen to disable dropout
        if self.freeze_cobra:
            self.cobra.eval()
        
        # Cast features to model dtype
        features = [f.to(dtype=next(self.cobra.parameters()).dtype) for f in features]
        if physical_positions is not None:
            physical_positions = physical_positions.to(device=features[0].device, dtype=torch.float32)
        embedding = self.cobra(
            features,
            seq_lengths=seq_lengths,
            physical_positions=physical_positions,
            fm_ids=self.fm_ids,
        )  # [B, embed_dim]
        
        # Cast embedding back to float32 for classifier
        embedding = embedding.float()
        
        logits = self.classifier(embedding)  # [B, num_classes]
        
        return {"logits": logits, "embedding": embedding}


class MultiViewClassifier(nn.Module):
    """
    Multi-view classifier with attention-based aggregation.
    
    Architecture:
        For each view plane: slice-level features (list of K tensors) -> COBRA -> volume-level embedding
        Attention aggregation: [emb_axial, emb_coronal, emb_sagittal] -> weighted sum -> aggregated embedding
        Classification: aggregated embedding -> classifier -> logits
    
    The attention weights provide interpretability by showing which view plane
    contributed most to the final prediction.
    """
    def __init__(
        self,
        cobra_model: Cobra,
        view_planes: List[str],
        input_dim: int,
        num_classes: int,
        classifier_hidden_dim: int = 512,
        classifier_dropout: float = 0.5,
        attention_hidden_dim: int = 128,
        attention_dropout: float = 0.1,
        freeze_cobra: bool = True,
        fm_ids: Optional[List[int]] = None,
    ):
        super().__init__()
        self.view_planes = view_planes
        self.num_views = len(view_planes)
        self.num_classes = num_classes
        self.freeze_cobra = freeze_cobra
        fm_id_tensor = None if fm_ids is None else torch.as_tensor(fm_ids, dtype=torch.long)
        self.register_buffer("fm_ids", fm_id_tensor, persistent=False)
        
        # COBRA encoder (shared across views, typically frozen)
        self.cobra = cobra_model
        if freeze_cobra:
            for param in self.cobra.parameters():
                param.requires_grad = False
        
        # Attention-based view aggregation
        self.attention_aggregator = MultiViewAttentionAggregator(
            input_dim=input_dim,
            hidden_dim=attention_hidden_dim,
            dropout=attention_dropout,
        )
        
        # Single classifier on aggregated embedding
        self.classifier = ClassifierHead(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=classifier_hidden_dim,
            dropout=classifier_dropout,
        )
    
    def forward(
        self,
        features: Dict[str, List[torch.Tensor]],
        seq_lengths: Dict[str, torch.Tensor],
        physical_positions: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: {view_plane: List of K tensors}. 
                      Global-only caches have shape [B, num_slices, input_embed_dim].
                      Tiled multi-crop CLS caches have shape [B, num_slices, num_tiled_regions, input_embed_dim].
            seq_lengths: {view_plane: [B]}
            physical_positions: {view_plane: [B, max_seq_len]}, used when Cobra has physical PE enabled.
        
        Returns:
            Dict with logits, attention_weights, and view_embeddings
        """
        # Keep COBRA in eval mode when frozen to disable dropout
        if self.freeze_cobra:
            self.cobra.eval()
        
        view_embeddings = {}
        
        # Get COBRA embeddings for each view
        for plane in self.view_planes:
            # Cast features to COBRA's dtype
            feats = [f.to(dtype=next(self.cobra.parameters()).dtype) for f in features[plane]]  # List of K tensors
            seq_length = seq_lengths[plane]
            phys_pos = physical_positions[plane] if physical_positions is not None else None
            if phys_pos is not None:
                phys_pos = phys_pos.to(device=feats[0].device, dtype=torch.float32)
            emb = self.cobra(
                feats,
                seq_lengths=seq_length,
                physical_positions=phys_pos,
                fm_ids=self.fm_ids,
            )  # [B, embed_dim]
            
            # Cast embedding back to float32 for downstream classifier
            view_embeddings[plane] = emb.float()

        # Stack embeddings: [B, num_view_planes, embed_dim]
        stacked_embeddings = torch.stack(list(view_embeddings.values()), dim=1)
        
        # Attention aggregation: [B, num_view_planes, embed_dim] -> [B, embed_dim]
        aggregated_emb, attention_weights = self.attention_aggregator(stacked_embeddings)
        
        # Classification: [B, embed_dim] -> [B, num_classes]
        logits = self.classifier(aggregated_emb)
        
        return {"logits": logits, "attention_weights": attention_weights, "view_embeddings": view_embeddings, "embedding": aggregated_emb}


# =============================================================================
# Utility Functions
# =============================================================================
def _log_trainable_params(model: nn.Module, accelerator: Accelerator):
    """Log total and trainable parameter counts."""
    if not accelerator.is_main_process:
        return
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    logger.info(
        f"Parameters: {trainable:,} trainable / {frozen:,} frozen / {total:,} total "
        f"({100 * trainable / total:.1f}% trainable)"
    )


def _build_optimizer(
    model: nn.Module,
    hyperparams: Dict,
    accelerator: Accelerator,
) -> torch.optim.Optimizer:
    """
    Build AdamW optimizer with optional differential learning rates for fine-tuning.

    When COBRA is frozen (linear probing):
        Single LR for all trainable params (classifier head only).

    When COBRA is unfrozen (fine-tuning):
        - base_lr for new / randomly-initialised modules:
            classifier head, attention_aggregator (multi-view only).
        - base_lr * backbone_lr_scale for the pretrained COBRA backbone.

    Args:
        model: SingleViewClassifier or MultiViewClassifier
        hyperparams: hyperparams section from config
        accelerator: HuggingFace Accelerator (for gated logging)

    Returns:
        Configured AdamW optimizer
    """
    base_lr = float(hyperparams.get("base_lr", 1e-5))
    weight_decay = float(hyperparams.get("weight_decay", 1e-3))

    # Linear probing: COBRA frozen
    if model.freeze_cobra:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(trainable_params, lr=base_lr, weight_decay=weight_decay)

    # Fine-tuning: differential learning rates
    backbone_lr_scale = float(hyperparams.get("backbone_lr_scale", 0.1))
    backbone_lr = base_lr * backbone_lr_scale

    # Pretrained COBRA params: base_lr * backbone_lr_scale. 
    pretrained_param_ids = set()
    pretrained_params = []
    for name, param in model.cobra.named_parameters():
        if not param.requires_grad:
            continue
        pretrained_params.append(param)
        pretrained_param_ids.add(id(param))

    # Everything else: base_lr (classifier head, attention_aggregator for multi-view)
    new_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in pretrained_param_ids
    ]

    param_groups = []
    if new_params:
        param_groups.append({"params": new_params, "lr": base_lr})
    if pretrained_params:
        param_groups.append({"params": pretrained_params, "lr": backbone_lr})
        
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


def _compute_loss(logits: torch.Tensor, labels: torch.Tensor, task: str, criterion: nn.Module) -> torch.Tensor:
    """Compute task-specific loss."""
    if task == "binary":
        return criterion(logits.squeeze(-1), labels.float())
    if task == "multilabel":
        return criterion(logits, labels.float())
    return criterion(logits, labels)


def _compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    task: str,
    num_classes: int,
    device: torch.device,
) -> Dict[str, float]:
    """Compute metrics for training/validation monitoring."""
    metrics = get_eval_metrics(task, num_classes, device)
    logits = logits.to(device)
    labels = labels.to(device)

    if task == "binary":
        labels = labels.long()
        if logits.ndim > 1:
            logits = logits.squeeze(-1)
        probs = torch.sigmoid(logits)
    elif task == "multilabel":
        labels = labels.long()
        probs = torch.sigmoid(logits)
    else:
        probs = torch.softmax(logits, dim=-1)

    for metric in metrics.values():
        metric.update(probs, labels)

    return {name: metric.compute().item() for name, metric in metrics.items()}


def _compute_predictions_and_metrics(
    all_probs: np.ndarray,
    all_true: np.ndarray,
    sample_ids: List,
    task: str,
    target_labels: List[str],
    output_dir: Optional[str] = None,
    class_names: Optional[List[str]] = None,
    thresholds: Optional[np.ndarray | float] = None,
) -> Dict:
    """Convert probabilities to prediction labels, compute metrics, and create results DataFrame.
    
    Args:
        all_probs: Predicted probabilities
        all_true: Ground truth labels
        sample_ids: Sample identifiers
        task: Task type (binary, multiclass, multilabel)
        target_labels: Column names used as classification targets
        output_dir: Optional output directory for visualization
        class_names: Optional list of class names for multiclass visualization.
                     If None and task is multiclass, auto-generates as ["class_0", "class_1", ...].
    """
    if task == "binary":
        all_probs = all_probs.squeeze(-1) if all_probs.ndim > 1 else all_probs
        threshold = 0.5 if thresholds is None else float(thresholds)
        all_preds = (all_probs >= threshold).astype(int)
    elif task == "multilabel":
        threshold_arr = (
            np.full(all_probs.shape[1], 0.5)
            if thresholds is None
            else np.asarray(thresholds, dtype=float)
        )
        all_preds = (all_probs >= threshold_arr).astype(int)
    else:
        all_preds = all_probs.argmax(axis=-1)
    
    # Compute metrics 
    viz_metrics = None
    if output_dir is not None:
        fig_dir = os.path.join(output_dir, "fig")

        # Determine visualization labels
        if task == "binary":
            viz_labels = target_labels[:1]
        elif task == "multiclass":
            if class_names is not None:
                viz_labels = class_names
            else:
                num_classes = all_probs.shape[-1]
                viz_labels = [f"class_{i}" for i in range(num_classes)]
        else:
            viz_labels = target_labels
        
        viz_metrics = compute_and_visualize_metrics(
            y_true=all_true,
            y_pred_prob=all_probs,
            task=task,
            class_labels=viz_labels,
            output_dir=fig_dir,
            thresholds=thresholds,
        )
        
        logger.info(f"Visualization plots saved to: {fig_dir}")
    
    # Create results DataFrame
    if task == "binary":
        df_results = pd.DataFrame({
            "exam_id": sample_ids,
            "true_labels": all_true.tolist(),
            "pred_labels": all_preds.tolist(),
            "pred_probs": all_probs.tolist()
        })
    elif task == "multilabel":
        df_results = pd.DataFrame({"exam_id": sample_ids})
        df_results["true_labels"] = [list(map(int, row)) for row in all_true]
        df_results["pred_labels"] = [[target_labels[i] for i, p in enumerate(row) if p >= 0.5] for row in all_probs]
        df_results["pred_probs"] = [list(row) for row in all_probs]
    else:
        df_results = pd.DataFrame({
            "exam_id": sample_ids,
            "true_labels": all_true.tolist(),
            "pred_labels": all_preds.tolist(),
            "pred_probs": [list(row) for row in all_probs]
        })
    
    return {"predictions": df_results, "metrics": viz_metrics}


def prepare_saving_metrics(metrics: Dict, task: str) -> Dict:
    """Prepare metrics dictionary for JSON serialization."""
    save_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, (np.floating, np.integer)):
            save_metrics[key] = float(value)
        elif isinstance(value, np.ndarray):
            save_metrics[key] = value.tolist()
        elif isinstance(value, dict):
            save_metrics[key] = prepare_saving_metrics(value, task)
        else:
            save_metrics[key] = value
    return save_metrics

def train_per_epoch(model: nn.Module, 
                    train_loader: DataLoader, 
                    optimizer: torch.optim.Optimizer, 
                    scheduler: torch.optim.lr_scheduler.LRScheduler, 
                    task: str, 
                    criterion: nn.Module, 
                    num_classes: int, 
                    accelerator: Accelerator) -> Tuple[float, Dict]:
    """Train the model for one epoch."""
    model.train()
    train_loss = 0.0
    train_logits_list = []
    train_labels_list = []
    for batch in train_loader:
        optimizer.zero_grad()
        with accelerator.autocast():
            outputs = model(batch["features"], batch["seq_lengths"], batch.get("physical_positions"))
            loss = _compute_loss(outputs["logits"], batch["labels"], task, criterion)
        accelerator.backward(loss)
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()
        train_logits_list.append(outputs["logits"].detach())
        train_labels_list.append(batch["labels"].detach())
    avg_train_loss = train_loss / len(train_loader)
    # Gather from all processes for metrics computation
    train_logits = accelerator.gather_for_metrics(torch.cat(train_logits_list, dim=0))
    train_labels = accelerator.gather_for_metrics(torch.cat(train_labels_list, dim=0))
    train_metrics = _compute_metrics(train_logits, train_labels, task, num_classes, accelerator.device)
    return avg_train_loss, train_metrics

def eval_per_epoch(model: nn.Module, 
                   val_loader: DataLoader, 
                   task: str, 
                   criterion: nn.Module, 
                   num_classes: int, 
                   accelerator: Accelerator,
                   multi_view: bool = False) -> Tuple[float, Dict, Optional[torch.Tensor]]:
    """Evaluate the model for one epoch."""
    model.eval()
    val_loss = 0.0
    val_logits_list = []
    val_labels_list = []
    if multi_view:
        val_attention_list = []
        
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(batch["features"], batch["seq_lengths"], batch.get("physical_positions"))
            loss = _compute_loss(outputs["logits"], batch["labels"], task, criterion)
            val_loss += loss.item()
            val_logits_list.append(outputs["logits"].detach())
            val_labels_list.append(batch["labels"].detach())
            if multi_view:
                val_attention_list.append(outputs["attention_weights"].detach())
    avg_val_loss = val_loss / len(val_loader)
    
    # Gather from all processes for metrics computation
    val_logits = accelerator.gather_for_metrics(torch.cat(val_logits_list, dim=0))
    val_labels = accelerator.gather_for_metrics(torch.cat(val_labels_list, dim=0))
    val_metrics = _compute_metrics(val_logits, val_labels, task, num_classes, accelerator.device)
    
    # Attention weights for multi-view
    val_attentions = None
    if multi_view and len(val_attention_list) > 0:
        val_attentions = accelerator.gather_for_metrics(torch.cat(val_attention_list, dim=0))
    
    return avg_val_loss, val_metrics, val_attentions
        
# =============================================================================
# Single-View Training and Evaluation
# =============================================================================
def train_single_view_classifier(
    model: SingleViewClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Dict,
    accelerator: Accelerator,
    output_dir: str,
    class_weights: Optional[torch.Tensor] = None,
    view_prefix: Optional[str] = None,
) -> Dict:
    """
    Train single-view classifier.
    
    Args:
        model: SingleViewClassifier
        train_loader: Training dataloader
        val_loader: Validation dataloader
        cfg: Configuration dict
        accelerator: HuggingFace Accelerator
        output_dir: Output directory for checkpoints
        class_weights: Optional class weights for loss
        view_prefix: Optional prefix for wandb logging (e.g., "axial", "coronal") to separate plots in multi-view evaluation
    
    Returns:
        Dict with best_model, best_val_auroc
    """
    hyperparams = cfg["hyperparams"]
    task = cfg["task"]
    
    # Setup optimizer (with different learning rates when fine-tuning)
    optimizer = _build_optimizer(model, hyperparams, accelerator)
    
    # Setup scheduler
    max_epochs = hyperparams.get("max_epochs", 100)
    num_training_steps = len(train_loader) * max_epochs
    warmup_ratio = hyperparams.get("warmup_ratio", 0.1)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(warmup_ratio * num_training_steps),
        num_training_steps=num_training_steps,
    )
    
    # Setup loss
    criterion = get_loss_criterion(task, class_weights)
    
    # Prepare with accelerator
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    
    # Early stopping
    patience = hyperparams.get("patience", None)
    ckpt_path = os.path.join(output_dir, "checkpoints")
    early_stopping = EarlyStopping(patience=patience, mode="max", ckpt_path=ckpt_path, accelerator=accelerator) if patience else None
    
    if accelerator.is_main_process:
        if patience:
            logger.info(f"Early stopping enabled with patience={patience}")
    
    best_val_auroc = 0.0
    best_state_dict = None
    num_classes = accelerator.unwrap_model(model).num_classes
    
    for epoch in tqdm(range(max_epochs), desc="Training classifier...", disable=not accelerator.is_main_process):
        # Training
        avg_train_loss, train_metrics = train_per_epoch(model, train_loader, optimizer, scheduler, task, criterion, num_classes, accelerator)
        
        # Validation
        avg_val_loss, val_metrics, _ = eval_per_epoch(model, val_loader, task, criterion, num_classes, accelerator, multi_view=False)
        val_auroc = val_metrics.get("auroc", 0.0)
        
        # Log to wandb
        if accelerator.is_main_process:
            current_lr = scheduler.get_last_lr()[0]
            
            # Add view prefix for multi-view logging to separate plots
            prefix = f"{view_prefix}/" if view_prefix else ""
            
            log_dict = {
                f"{prefix}lr": current_lr,
                f"{prefix}train/loss_per_epoch": avg_train_loss,
                f"{prefix}val/loss_per_epoch": avg_val_loss,
                f"{prefix}epoch": epoch + 1,
            }
            for name, value in train_metrics.items():
                log_dict[f"{prefix}train/{name}"] = value
            for name, value in val_metrics.items():
                log_dict[f"{prefix}val/{name}"] = value
            wandb.log(log_dict)
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, val_auroc={val_auroc:.4f}")
        
        # Save best model
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state_dict = accelerator.get_state_dict(model)
            
            if accelerator.is_main_process:
                ckpt_dir = os.path.join(output_dir, "ckpt")
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(best_state_dict, os.path.join(ckpt_dir, "classifier.pt"))
        
        # Early stopping
        if early_stopping:
            should_stop, best_score = early_stopping.step(
                                            val_score=val_auroc,
                                            model=model,
                                            optimizer=optimizer,
                                            scheduler=scheduler,
                                            epoch=epoch
                                        )
            if should_stop:
                if accelerator.is_main_process:
                    logger.info(f"Early stopping at epoch {epoch} with best validation AUROC {best_score:.4f}.")
                # Load best model on all processes
                best_model = early_stopping.load_best_model(accelerator.unwrap_model(model))
                return {"best_model": best_model, "best_val_auroc": best_score}
    
    if accelerator.is_main_process:
        logger.info(f"Training complete. Best validation AUROC: {best_val_auroc:.4f}")
    
    # Load best weights
    if best_state_dict:
        accelerator.unwrap_model(model).load_state_dict(best_state_dict)
    
    return {"best_model": accelerator.unwrap_model(model), "best_val_auroc": best_val_auroc}


def evaluate_single_view_classifier(
    model: SingleViewClassifier,
    test_loader: DataLoader,
    cfg: Dict,
    accelerator: Accelerator,
    output_dir: str,
    track_emissions: bool = True,
    val_threshold_loader: Optional[DataLoader] = None,
) -> Dict:
    """Evaluate on test set, optionally using validation data to derive thresholds."""
    task = cfg["task"]
    target_labels = cfg["target_labels"]
    
    # Prepare model and dataloader for distributed evaluation
    if val_threshold_loader is not None:
        model, test_loader, val_threshold_loader = accelerator.prepare(model, test_loader, val_threshold_loader)
    else:
        model, test_loader = accelerator.prepare(model, test_loader)
    model.eval()

    thresholds = None
    if val_threshold_loader is not None:
        val_logits_list = []
        val_labels_list = []
        with torch.no_grad():
            for batch in tqdm(
                val_threshold_loader,
                desc="Collecting validation thresholds...",
                disable=not accelerator.is_main_process,
            ):
                outputs = model(batch["features"], batch["seq_lengths"], batch.get("physical_positions"))
                val_logits_list.append(outputs["logits"].detach())
                val_labels_list.append(batch["labels"].detach())
        val_logits = accelerator.gather_for_metrics(torch.cat(val_logits_list, dim=0))
        val_labels = accelerator.gather_for_metrics(torch.cat(val_labels_list, dim=0))
        if accelerator.is_main_process:
            if task == "binary":
                val_probs = torch.sigmoid(val_logits.squeeze(-1)).cpu().numpy()
            elif task == "multilabel":
                val_probs = torch.sigmoid(val_logits).cpu().numpy()
            else:
                val_probs = F.softmax(val_logits, dim=-1).cpu().numpy()
            thresholds = compute_youden_thresholds(
                y_true=val_labels.cpu().numpy(),
                y_pred_prob=val_probs,
                task=task,
            )
    
    tracker = None
    if track_emissions and accelerator.is_main_process:
        tracker = EmissionsTracker(
            project_name="medslim-inference",
            output_dir=output_dir,
            log_level="warning",
        )
        tracker.start()
    
    all_logits = []
    all_labels = []
    all_sample_ids = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating...", disable=not accelerator.is_main_process):
            outputs = model(batch["features"], batch["seq_lengths"], batch.get("physical_positions"))
            all_logits.append(outputs["logits"].detach())
            all_labels.append(batch["labels"].detach())
            all_sample_ids.extend(batch["sample_ids"])
    
    if tracker is not None:
        emissions = tracker.stop()
        logger.info(f"Inference emissions: {emissions:.6f} kg CO2eq")
    
    # Gather predictions from all processes
    all_logits = accelerator.gather_for_metrics(torch.cat(all_logits, dim=0))
    all_labels = accelerator.gather_for_metrics(torch.cat(all_labels, dim=0))
    
    # Only compute metrics on main process
    if not accelerator.is_main_process:
        return {"predictions": None, "metrics": None}
    
    # Convert logits to probabilities
    if task == "binary":
        all_probs = torch.sigmoid(all_logits.squeeze(-1)).cpu().numpy()
    elif task == "multilabel":
        all_probs = torch.sigmoid(all_logits).cpu().numpy()
    else:
        all_probs = F.softmax(all_logits, dim=-1).cpu().numpy()
    
    eval_results = _compute_predictions_and_metrics(
        all_probs=all_probs,
        all_true=all_labels.cpu().numpy(),
        sample_ids=all_sample_ids,
        task=task,
        target_labels=target_labels,
        output_dir=output_dir,
        class_names=cfg.get("class_names"),
        thresholds=thresholds,
    )
    
    return eval_results


def run_single_view_evaluation(
    cobra_model: Cobra,
    train_dataset: FeatClassificationDataset,
    test_dataset: FeatClassificationDataset,
    cfg: Dict,
    accelerator: Accelerator,
    output_dir: str,
    class_weights: Optional[torch.Tensor] = None,
    val_dataset: Optional[FeatClassificationDataset] = None,
    train_fraction: float = 1.0,
    subsample_seed: int = 42,
) -> Dict:
    """
    Run complete single-view evaluation pipeline.
    
    1. Create train/val split (or val_dataset if the validation set is already provided by the data provider)
    2. Optionally subsample the training set to `train_fraction` (few-shot)
    3. Initialize SingleViewClassifier (COBRA + classifier head)
    4. Linear probe training
    5. Evaluate on test set
    """
    hyperparams = cfg["hyperparams"]
    linear_hyperparams = hyperparams.get("linear", hyperparams)
    task = cfg["task"]
    
    if accelerator.is_main_process:
        logger.info(f"\n{'='*50}")
        logger.info("Single-view evaluation")
        logger.info(f"{'='*50}")
    
    # Create train/val split or use dedicated val set.
    # `train_candidate_idx` are the positional indices eligible for training
    # (i.e. everything except the validation split). Few-shot subsampling is
    # applied only to these candidates, never to validation/test.
    if val_dataset is not None:
        train_candidate_idx = np.arange(len(train_dataset))
        val_data = val_dataset
    else:
        all_labels = np.array([train_dataset._get_label(sid).numpy() for sid in train_dataset.sample_ids])
        val_split_ratio = hyperparams.get("val_split_ratio", 0.1)
        
        if task == "multilabel":
            n_splits = max(2, int(1.0 / val_split_ratio))
            kfold = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=subsample_seed)
            splits = list(kfold.split(X=np.arange(len(train_dataset)), y=all_labels))
            train_idx, val_idx = splits[0]
        else:
            train_idx, val_idx = train_test_split(
                np.arange(len(train_dataset)),
                test_size=val_split_ratio,
                stratify=all_labels,
                shuffle=True,
                random_state=subsample_seed,
            )
        
        train_candidate_idx = train_idx
        val_data = Subset(train_dataset, val_idx)

    # Few-shot subsampling with subset-aware class weights
    selected_train_idx, class_weights = prepare_training_indices(
        dataset=train_dataset,
        candidate_idx=train_candidate_idx,
        cfg=cfg,
        accelerator=accelerator,
        train_fraction=train_fraction,
        subsample_seed=subsample_seed,
        context="single-view",
    )
    train_data = Subset(train_dataset, selected_train_idx.tolist())

    if accelerator.is_main_process:
        logger.info(f"Training samples:  {len(train_data)}, Validation samples: {len(val_data)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=hyperparams["batch_size"], shuffle=True, 
                              collate_fn=linear_classifier_collate_fn, num_workers=hyperparams.get("num_workers", 0))
    val_loader = DataLoader(val_data, batch_size=hyperparams["batch_size"], shuffle=False,
                            collate_fn=linear_classifier_collate_fn, num_workers=hyperparams.get("num_workers", 0))
    test_loader = DataLoader(test_dataset, batch_size=hyperparams["batch_size"], shuffle=False,
                             collate_fn=linear_classifier_collate_fn, num_workers=hyperparams.get("num_workers", 0))
    
    # Get dimensions
    num_classes = get_num_classes(task, cfg["target_labels"], train_dataset)
    input_dim = cobra_model.output_dim
    
    if accelerator.is_main_process:
        logger.info(f"Num classes: {num_classes}, COBRA output dim: {input_dim}")
    
    # Initialize model
    freeze_cobra = cfg.get("freeze_cobra", True)
    model = SingleViewClassifier(
        cobra_model=cobra_model,
        input_dim=input_dim,
        num_classes=num_classes,
        classifier_hidden_dim=linear_hyperparams.get("hidden_dim", 512),
        classifier_dropout=linear_hyperparams.get("dropout", 0.5),
        freeze_cobra=freeze_cobra,
        fm_ids=cfg.get("eval_fm_ids"),
    )
    
    if accelerator.is_main_process:
        mode = "Linear Probing" if freeze_cobra else "Fine-tuning"
        logger.info(f"Training mode: {mode}")
    _log_trainable_params(model, accelerator)
    
    # Train
    training_results = train_single_view_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        accelerator=accelerator,
        output_dir=output_dir,
        class_weights=class_weights,
    )
    
    best_model = training_results["best_model"]
    
    if accelerator.is_main_process:
        logger.info(f"\n{'='*50}")
        logger.info("Evaluating on test set")
        logger.info(f"{'='*50}")
    
    # Evaluate
    eval_results = evaluate_single_view_classifier(
        model=best_model,
        test_loader=test_loader,
        cfg=cfg,
        accelerator=accelerator,
        output_dir=output_dir,
        val_threshold_loader=val_loader,
    )
    
    # Wait for all processes before saving
    accelerator.wait_for_everyone()
    
    # Save results
    if accelerator.is_main_process:
        table_dir = os.path.join(output_dir, "table")
        os.makedirs(table_dir, exist_ok=True)
        
        # Save predictions
        eval_results["predictions"].to_csv(os.path.join(table_dir, "predictions.csv"), index=False)
        
        # Save metrics
        if eval_results["metrics"]:
            save_metrics = prepare_saving_metrics(eval_results["metrics"], task)
            with open(os.path.join(table_dir, "metrics.json"), "w") as f:
                json.dump(save_metrics, f, indent=4)
            
            # Log to wandb
            auroc = save_metrics["AUROC"] if task == "binary" else save_metrics["overall"]["AUROC"]
            auprc = save_metrics["AUPRC"] if task == "binary" else save_metrics["overall"]["AUPRC"]
            wandb.run.summary["test_auroc"] = auroc
            wandb.run.summary["test_auprc"] = auprc
            
            logger.info(f"Test AUROC: {auroc:.4f}, Test AUPRC: {auprc:.4f}")
            eval_results["test_auroc"] = auroc
            eval_results["test_auprc"] = auprc
    
    return eval_results


def run_single_view_kfold_evaluation(
    cobra_model: Cobra,
    train_dataset: FeatClassificationDataset,
    test_dataset: FeatClassificationDataset,
    cfg: Dict,
    accelerator: Accelerator,
    output_dir: str,
    n_folds: int = 5,
    class_weights: Optional[torch.Tensor] = None,
    train_fraction: float = 1.0,
    subsample_seed: int = 42,
) -> Dict:
    """
    Run K-fold cross-validation for single-view evaluation.

    Args:
        cobra_model: Pretrained COBRA model.
        train_dataset: Training dataset.
        test_dataset: Test dataset.
        cfg: Configuration dict.
        accelerator: HuggingFace Accelerator.
        output_dir: Output directory.
        n_folds: Number of cross-validation folds.
        class_weights: Optional class weights for loss.
        train_fraction: Few-shot fraction applied to each fold's training split.
        subsample_seed: Seed driving fold splitting and per-fold subsampling.

    Returns:
        Dict with per-fold results, mean/std AUROC, and best-fold predictions.
    """
    hyperparams = cfg["hyperparams"]
    linear_hyperparams = hyperparams.get("linear", hyperparams)
    task = cfg["task"]

    if accelerator.is_main_process:
        logger.info(f"\n{'='*50}")
        logger.info(f"{n_folds}-Fold Cross-Validation Evaluation")
        logger.info(f"{'='*50}")

    all_labels = np.array([
        train_dataset._get_label(sid).numpy()
        for sid in train_dataset.sample_ids
    ])

    if task == "multilabel":
        kfold = MultilabelStratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=subsample_seed
        )
        fold_splits = list(kfold.split(
            X=np.arange(len(train_dataset)), y=all_labels
        ))
    else:
        kfold = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=subsample_seed
        )
        fold_splits = list(kfold.split(
            X=np.arange(len(train_dataset)), y=all_labels
        ))

    num_classes = get_num_classes(task, cfg["target_labels"], train_dataset)
    input_dim = cobra_model.output_dim
    freeze_cobra = cfg.get("freeze_cobra", True)

    fold_aurocs = []
    fold_auprcs = []
    fold_results = []
    best_fold_auroc = -1.0
    best_fold_eval = None

    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        if accelerator.is_main_process:
            logger.info(f"\n{'='*50}")
            logger.info(f"Fold {fold_idx + 1}/{n_folds}")
            logger.info(f"{'='*50}")

        # Few-shot subsampling applies only to this fold's training indices;
        # the fold validation split (val_idx) is left unchanged.
        selected_train_idx, class_weights = prepare_training_indices(
            dataset=train_dataset,
            candidate_idx=train_idx,
            cfg=cfg,
            accelerator=accelerator,
            train_fraction=train_fraction,
            subsample_seed=subsample_seed + fold_idx,
            context=f"fold {fold_idx + 1}",
        )
        train_data = Subset(train_dataset, selected_train_idx.tolist())
        val_data = Subset(train_dataset, val_idx.tolist())

        if accelerator.is_main_process:
            logger.info(
                f"Training: {len(train_data)}, Validation: {len(val_data)}"
            )

        train_loader = DataLoader(
            train_data,
            batch_size=hyperparams["batch_size"],
            shuffle=True,
            collate_fn=linear_classifier_collate_fn,
            num_workers=hyperparams.get("num_workers", 0),
        )
        val_loader = DataLoader(
            val_data,
            batch_size=hyperparams["batch_size"],
            shuffle=False,
            collate_fn=linear_classifier_collate_fn,
            num_workers=hyperparams.get("num_workers", 0),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=hyperparams["batch_size"],
            shuffle=False,
            collate_fn=linear_classifier_collate_fn,
            num_workers=hyperparams.get("num_workers", 0),
        )

        model = SingleViewClassifier(
            cobra_model=cobra_model,
            input_dim=input_dim,
            num_classes=num_classes,
            classifier_hidden_dim=linear_hyperparams.get("hidden_dim", 512),
            classifier_dropout=linear_hyperparams.get("dropout", 0.5),
            freeze_cobra=freeze_cobra,
            fm_ids=cfg.get("eval_fm_ids"),
        )
        if fold_idx == 0:
            _log_trainable_params(model, accelerator)

        fold_output_dir = os.path.join(output_dir, f"fold_{fold_idx + 1}")
        if accelerator.is_main_process:
            os.makedirs(fold_output_dir, exist_ok=True)

        training_results = train_single_view_classifier(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg=cfg,
            accelerator=accelerator,
            output_dir=fold_output_dir,
            class_weights=class_weights,
            view_prefix=f"fold{fold_idx + 1}",
        )

        best_model = training_results["best_model"]

        eval_results = evaluate_single_view_classifier(
            model=best_model,
            test_loader=test_loader,
            cfg=cfg,
            accelerator=accelerator,
            output_dir=fold_output_dir,
            track_emissions=(fold_idx == 0),
            val_threshold_loader=val_loader,
        )

        accelerator.wait_for_everyone()

        if accelerator.is_main_process and eval_results["metrics"]:
            save_metrics = prepare_saving_metrics(eval_results["metrics"], task)
            auroc = (
                save_metrics["AUROC"]
                if task == "binary"
                else save_metrics["overall"]["AUROC"]
            )
            auprc = (
                save_metrics["AUPRC"]
                if task == "binary"
                else save_metrics["overall"]["AUPRC"]
            )

            fold_aurocs.append(auroc)
            fold_auprcs.append(auprc)
            fold_results.append(save_metrics)

            logger.info(
                f"Fold {fold_idx + 1}: AUROC={auroc:.4f}, AUPRC={auprc:.4f}"
            )

            wandb.log({
                f"fold{fold_idx + 1}/test_auroc": auroc,
                f"fold{fold_idx + 1}/test_auprc": auprc,
            })

            # Save fold metrics
            table_dir = os.path.join(fold_output_dir, "table")
            os.makedirs(table_dir, exist_ok=True)
            eval_results["predictions"].to_csv(
                os.path.join(table_dir, "predictions.csv"), index=False
            )
            with open(os.path.join(table_dir, "metrics.json"), "w") as f:
                json.dump(save_metrics, f, indent=4)

            if auroc > best_fold_auroc:
                best_fold_auroc = auroc
                best_fold_eval = eval_results

    # Aggregate results across folds
    if accelerator.is_main_process and fold_aurocs:
        mean_auroc = float(np.mean(fold_aurocs))
        std_auroc = float(np.std(fold_aurocs))
        mean_auprc = float(np.mean(fold_auprcs))
        std_auprc = float(np.std(fold_auprcs))

        logger.info(f"\n{'='*50}")
        logger.info(f"{n_folds}-Fold Cross-Validation Results")
        logger.info(f"{'='*50}")
        for i, (auc_val, prc_val) in enumerate(
            zip(fold_aurocs, fold_auprcs)
        ):
            logger.info(
                f"  Fold {i + 1}: AUROC={auc_val:.4f}, AUPRC={prc_val:.4f}"
            )
        logger.info(f"  Mean AUROC: {mean_auroc:.4f} +/- {std_auroc:.4f}")
        logger.info(f"  Mean AUPRC: {mean_auprc:.4f} +/- {std_auprc:.4f}")

        # Save aggregated results
        cv_summary = {
            "n_folds": n_folds,
            "fold_aurocs": fold_aurocs,
            "fold_auprcs": fold_auprcs,
            "mean_auroc": mean_auroc,
            "std_auroc": std_auroc,
            "mean_auprc": mean_auprc,
            "std_auprc": std_auprc,
        }
        table_dir = os.path.join(output_dir, "table")
        os.makedirs(table_dir, exist_ok=True)
        with open(os.path.join(table_dir, "cv_summary.json"), "w") as f:
            json.dump(cv_summary, f, indent=4)

        wandb.run.summary["cv_mean_auroc"] = mean_auroc
        wandb.run.summary["cv_std_auroc"] = std_auroc
        wandb.run.summary["cv_mean_auprc"] = mean_auprc
        wandb.run.summary["cv_std_auprc"] = std_auprc
        wandb.run.summary["test_auroc"] = mean_auroc
        wandb.run.summary["test_auprc"] = mean_auprc

        # Surface fold-mean scalars so repeat-level aggregation can consume them.
        if best_fold_eval is None:
            best_fold_eval = {}
        best_fold_eval["test_auroc"] = mean_auroc
        best_fold_eval["test_auprc"] = mean_auprc
        best_fold_eval["cv_mean_auroc"] = mean_auroc
        best_fold_eval["cv_std_auroc"] = std_auroc
        best_fold_eval["cv_mean_auprc"] = mean_auprc
        best_fold_eval["cv_std_auprc"] = std_auprc

    return best_fold_eval if best_fold_eval else {}


# =============================================================================
# Multi-View Logistic Regression Ensemble
# =============================================================================
def collect_predictions_per_view_classifier(
    cobra_model: Cobra,
    train_subset: Dataset,
    val_subset: Dataset,
    test_dataset: FeatClassificationDataset,
    view_plane: str,
    cfg: Dict,
    accelerator: Accelerator,
    output_dir: str,
    class_weights: Optional[torch.Tensor] = None,
) -> Dict:
    """
    Train a single-view classifier to get per-view predictions on validation and test sets for logistic regression ensemble.
    
    Args:
        train_subset: Training data (Subset when split from train, or full Dataset when using dedicated val set)
        val_subset: Validation data (Subset when split from train, or full Dataset when using dedicated val set)
    
    Returns:
        Dict with val_probs, val_labels, test_probs, test_labels, sample_ids
    """
    hyperparams = cfg["hyperparams"]
    linear_hyperparams = hyperparams.get("linear", hyperparams)
    task = cfg["task"]
    
    # Create dataloaders
    train_loader = DataLoader(
        train_subset, 
        batch_size=hyperparams["batch_size"], 
        shuffle=True,
        collate_fn=linear_classifier_collate_fn, 
        num_workers=hyperparams.get("num_workers", 0)
    )
    val_loader = DataLoader(
        val_subset, 
        batch_size=hyperparams["batch_size"], 
        shuffle=False,
        collate_fn=linear_classifier_collate_fn, 
        num_workers=hyperparams.get("num_workers", 0)
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=hyperparams["batch_size"], 
        shuffle=False,
        collate_fn=linear_classifier_collate_fn, 
        num_workers=hyperparams.get("num_workers", 0)
    )
    
    # Get dimensions
    num_classes = get_num_classes(task, cfg["target_labels"], test_dataset)
    input_dim = cobra_model.output_dim
    
    # Initialize model
    freeze_cobra = cfg.get("freeze_cobra", True)
    model = SingleViewClassifier(
        cobra_model=cobra_model,
        input_dim=input_dim,
        num_classes=num_classes,
        classifier_hidden_dim=linear_hyperparams.get("hidden_dim", 512),
        classifier_dropout=linear_hyperparams.get("dropout", 0.5),
        freeze_cobra=freeze_cobra,
        fm_ids=cfg.get("eval_fm_ids"),
    )
    _log_trainable_params(model, accelerator)
    
    # Create view-specific output directory
    view_output_dir = os.path.join(output_dir, f"{view_plane}")
    if accelerator.is_main_process:
        os.makedirs(view_output_dir, exist_ok=True)
    
    # Train with view prefix for multi-view wandb logging
    training_results = train_single_view_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        accelerator=accelerator,
        output_dir=view_output_dir,
        class_weights=class_weights,
        view_prefix=view_plane, 
    )
    
    best_model = training_results["best_model"]
    
    # Collect predictions on validation set
    best_model, val_loader = accelerator.prepare(best_model, val_loader)
    best_model.eval()
    
    val_logits_list = []
    val_labels_list = []
    val_sample_ids = []
    
    with torch.no_grad():
        for batch in val_loader:
            outputs = best_model(batch["features"], batch["seq_lengths"], batch.get("physical_positions"))
            val_logits_list.append(outputs["logits"].detach())
            val_labels_list.append(batch["labels"].detach())
            val_sample_ids.extend(batch["sample_ids"])
    
    val_logits = accelerator.gather_for_metrics(torch.cat(val_logits_list, dim=0))
    val_labels = accelerator.gather_for_metrics(torch.cat(val_labels_list, dim=0))
    
    # Collect predictions on test set
    test_loader = accelerator.prepare(test_loader)
    
    test_logits_list = []
    test_labels_list = []
    test_sample_ids = []
    
    with torch.no_grad():
        for batch in test_loader:
            outputs = best_model(batch["features"], batch["seq_lengths"], batch.get("physical_positions"))
            test_logits_list.append(outputs["logits"].detach())
            test_labels_list.append(batch["labels"].detach())
            test_sample_ids.extend(batch["sample_ids"])
    
    test_logits = accelerator.gather_for_metrics(torch.cat(test_logits_list, dim=0))
    test_labels = accelerator.gather_for_metrics(torch.cat(test_labels_list, dim=0))
    
    # Convert logits to probabilities
    if task == "binary":
        val_probs = torch.sigmoid(val_logits.squeeze(-1)).cpu().numpy()
        test_probs = torch.sigmoid(test_logits.squeeze(-1)).cpu().numpy()
    elif task == "multilabel":
        val_probs = torch.sigmoid(val_logits).cpu().numpy()
        test_probs = torch.sigmoid(test_logits).cpu().numpy()
    else:
        val_probs = F.softmax(val_logits, dim=-1).cpu().numpy()
        test_probs = F.softmax(test_logits, dim=-1).cpu().numpy()
    
    return {
        "val_probs": val_probs,
        "val_labels": val_labels.cpu().numpy(),
        "val_sample_ids": val_sample_ids,
        "test_probs": test_probs,
        "test_labels": test_labels.cpu().numpy(),
        "test_sample_ids": test_sample_ids,
        "best_val_auroc": training_results["best_val_auroc"],
    }


def train_logistic_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    task: str,
) -> LogisticRegression or List[LogisticRegression]:
    """
    Train logistic regression ensemble on stacked view predictions.
    
    Args:
        X_train: Stacked predictions [N, num_views] for binary, [N, num_views * num_classes] for multiclass
        y_train: Labels [N] or [N, num_classes] for multilabel
        task: Task type (binary, multiclass, multilabel)
    
    Returns:
        Trained LogisticRegression model(s)
    """
    if task == "multilabel":
        # For multilabel, train separate binary logistic regression for each label
        # Each label gets probabilities from all views: [N, num_views]
        num_labels = y_train.shape[1]
        if X_train.shape[1] % num_labels != 0:
            raise ValueError(f"X_train shape {X_train.shape} should be divisible by num_labels {num_labels}")
        num_views = X_train.shape[1] // num_labels
        ensembles = []
        for label_idx in range(num_labels):
            # Extract probabilities for this label from all views
            # X_train shape: [N, num_views * num_labels] with order [view1_label1, ..., view1_labelN, view2_label1, ..., view2_labelN, view3_label1, ..., view3_labelN]
            # We need to extract columns [label_idx, num_labels + label_idx, 2 * num_labels + label_idx]
            X_label = np.column_stack([
                X_train[:, view_idx * num_labels + label_idx] 
                for view_idx in range(num_views)
            ])
            
            # Binary labels for this specific label
            y_label = y_train[:, label_idx]
            
            # Train binary logistic regression for this label
            ensemble = LogisticRegression(
                solver="lbfgs",
                max_iter=1000,
                random_state=42,
                class_weight="balanced", # Handle class imbalance
            )
            ensemble.fit(X_label, y_label)
            ensembles.append(ensemble)
        
        return ensembles
    
    # Binary or multiclass: single logistic regression
    ensemble = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
        class_weight="balanced",  # Handle class imbalance
    )
    ensemble.fit(X_train, y_train)
    
    return ensemble


def predict_logistic_ensemble_proba(
    ensemble: LogisticRegression or List[LogisticRegression],
    X: np.ndarray,
    task: str,
    target_labels: List[str],
    view_planes: List[str],
) -> np.ndarray:
    """Predict probabilities from a logistic ensemble for binary/multilabel/multiclass tasks."""
    if task == "binary":
        return ensemble.predict_proba(X)[:, 1]
    if task == "multilabel":
        num_labels = len(target_labels)
        num_views = len(view_planes)
        y_prob_list = []
        for label_idx, ensemble_label in enumerate(ensemble):
            X_label = np.column_stack([
                X[:, view_idx * num_labels + label_idx]
                for view_idx in range(num_views)
            ])
            y_prob_list.append(ensemble_label.predict_proba(X_label)[:, 1])
        return np.column_stack(y_prob_list)
    return ensemble.predict_proba(X)


def evaluate_logistic_ensemble(
    ensemble: LogisticRegression or List[LogisticRegression],
    X: np.ndarray,
    y: np.ndarray,
    sample_ids: List,
    task: str,
    target_labels: List[str],
    view_planes: List[str],
    output_dir: Optional[str] = None,
    class_names: Optional[List[str]] = None,
    thresholds: Optional[np.ndarray | float] = None,
) -> Dict:
    """
    Evaluate logistic regression ensemble and return metrics.
    
    Args:
        ensemble: LogisticRegression model or a list of LogisticRegression models for multilabel
        X: Stacked predictions
        y: Labels
        task: Task type
        target_labels: List of label names
        view_planes: List of view plane names
        output_dir: Optional output directory
        class_names: Optional list of human-readable class names for multiclass visualization
    
    Returns:
        Dict with predictions DataFrame, metrics, and ensemble weights
    """
    # Get predictions
    y_prob = predict_logistic_ensemble_proba(
        ensemble=ensemble,
        X=X,
        task=task,
        target_labels=target_labels,
        view_planes=view_planes,
    )
    
    # Compute predictions and metrics
    eval_results = _compute_predictions_and_metrics(
        all_probs=y_prob,
        all_true=y,
        sample_ids=sample_ids,
        task=task,
        target_labels=target_labels,
        output_dir=output_dir,
        class_names=class_names,
        thresholds=thresholds,
    )
    
    # Add ensemble weights for interpretability
    if task == "binary":
        # Binary: single coefficient per view
        ensemble_weights = {
            plane: float(ensemble.coef_[0][i]) for i, plane in enumerate(view_planes)
        }
        ensemble_weights["intercept"] = float(ensemble.intercept_[0])
    elif task == "multilabel":
        # Multilabel: separate weights per label
        num_labels = len(target_labels)
        num_views = len(view_planes)
        ensemble_weights = {}
        
        for label_idx, (label_name, ensemble_label) in enumerate(zip(target_labels, ensemble)):
            label_weights = {
                plane: float(ensemble_label.coef_[0][view_idx]) 
                for view_idx, plane in enumerate(view_planes)
            }
            label_weights["intercept"] = float(ensemble_label.intercept_[0])
            ensemble_weights[label_name] = label_weights
    else:
        # Multiclass: coefficients per class
        num_classes = len(target_labels)
        num_views = len(view_planes)
        ensemble_weights = {}
        for class_idx, class_name in enumerate(target_labels):
            class_weights = {}
            for view_idx, plane in enumerate(view_planes):
                # For multiclass, features are concatenated: [view1_class1, view1_class2, ..., view2_class1, ...]
                feature_idx = view_idx * num_classes + class_idx
                class_weights[plane] = float(ensemble.coef_[class_idx][feature_idx])
            class_weights["intercept"] = float(ensemble.intercept_[class_idx])
            ensemble_weights[class_name] = class_weights
    
    eval_results["ensemble_weights"] = ensemble_weights
    
    return eval_results


def run_multiview_logistic_ensemble(
    cobra_model: Cobra,
    train_datasets: Dict[str, FeatClassificationDataset],
    test_datasets: Dict[str, FeatClassificationDataset],
    cfg: Dict,
    accelerator: Accelerator,
    output_dir: str,
    view_planes: List[str],
    class_weights: Optional[torch.Tensor] = None,
    val_datasets: Optional[Dict[str, FeatClassificationDataset]] = None,
    train_fraction: float = 1.0,
    subsample_seed: int = 42,
) -> Dict:
    """
    Run multi-view evaluation using logistic regression ensemble.
    
    Pipeline:
    1. For each view plane:
       - Train a single-view classifier
       - Collect predictions on validation and test sets
    2. Stack predictions from all views: X = [prob_view1, prob_view2, ...]
    3. Fit LogisticRegression on validation predictions
    4. Evaluate ensemble on test set
    
    Args:
        cobra_model: Pretrained COBRA model
        train_datasets: Dict mapping view_plane -> FeatClassificationDataset for training
        test_datasets: Dict mapping view_plane -> FeatClassificationDataset for testing
        cfg: Configuration dict
        accelerator: HuggingFace Accelerator
        output_dir: Output directory
        view_planes: List of view plane names
        class_weights: Optional class weights
        val_datasets: Optional dict mapping view_plane -> FeatClassificationDataset for validation.
                      If provided, uses dedicated val sets instead of splitting train.
    """
    hyperparams = cfg["hyperparams"]
    task = cfg["task"]
    target_labels = cfg["target_labels"]
    
    if accelerator.is_main_process:
        logger.info(f"\n{'='*50}")
        logger.info("Multi-view Logistic Regression Ensemble")
        logger.info(f"View planes: {view_planes}")
        logger.info(f"{'='*50}")
    
    # Create consistent train/val split or use dedicated val sets.
    # The split (and any few-shot subsampling) is computed once on the reference
    # channel so that selected indices stay aligned across all views/sequences.
    use_dedicated_val = val_datasets is not None
    ref_dataset = train_datasets[view_planes[0]]

    if use_dedicated_val:
        train_candidate_idx = np.arange(len(ref_dataset))
        val_idx = None
        if accelerator.is_main_process:
            n_train = len(ref_dataset)
            n_val = len(val_datasets[view_planes[0]])
            logger.info(f"Using dedicated validation set. Training: {n_train}, Validation: {n_val}")
    else:
        all_labels = np.array([ref_dataset._get_label(sid).numpy() for sid in ref_dataset.sample_ids])
        val_split_ratio = hyperparams.get("val_split_ratio", 0.1)
        
        if task == "multilabel":
            n_splits = max(2, int(1.0 / val_split_ratio))
            kfold = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=subsample_seed)
            splits = list(kfold.split(X=np.arange(len(ref_dataset)), y=all_labels))
            train_idx, val_idx = splits[0]
        else:
            train_idx, val_idx = train_test_split(
                np.arange(len(ref_dataset)),
                test_size=val_split_ratio,
                stratify=all_labels,
                random_state=subsample_seed,
            )
        train_candidate_idx = train_idx
        if accelerator.is_main_process:
            logger.info(f"Split training set. Training: {len(train_idx)}, Validation: {len(val_idx)}")

    # Few-shot subsampling with subset-aware class weights
    selected_train_idx, class_weights = prepare_training_indices(
        dataset=ref_dataset,
        candidate_idx=train_candidate_idx,
        cfg=cfg,
        accelerator=accelerator,
        train_fraction=train_fraction,
        subsample_seed=subsample_seed,
        context="multiview-ensemble",
    )
    selected_train_idx = selected_train_idx.tolist()

    # Step 1: Train single-view classifiers and collect predictions for each view
    view_predictions = {}
    
    for plane in view_planes:
        if accelerator.is_main_process:
            logger.info(f"\n{'='*50}")
            logger.info(f"Training single-view classifier for: {plane}")
            logger.info(f"{'='*50}")
        
        train_dataset_view_plane = train_datasets[plane]
        test_dataset_view_plane = test_datasets[plane]
        
        # Use dedicated val set or split from training set
        train_subset = Subset(train_dataset_view_plane, selected_train_idx)
        if use_dedicated_val:
            val_subset = val_datasets[plane]
        else:
            val_subset = Subset(train_dataset_view_plane, val_idx)
        
        # Train and collect predictions
        predictions = collect_predictions_per_view_classifier(
            cobra_model=cobra_model,
            train_subset=train_subset,
            val_subset=val_subset,
            test_dataset=test_dataset_view_plane,
            view_plane=plane,
            cfg=cfg,
            accelerator=accelerator,
            output_dir=output_dir,
            class_weights=class_weights,
        )
        
        view_predictions[plane] = predictions
        
        if accelerator.is_main_process:
            logger.info(f"{plane} view - Val AUROC: {predictions['best_val_auroc']:.4f}")
    
    # Only proceed on main process for ensemble fitting
    if not accelerator.is_main_process:
        return {"predictions": None, "metrics": None, "ensemble_weights": None}
    
    # Step 2: Stack predictions
    # For binary: X = [prob_view1, prob_view2, prob_view3] -> shape [N, num_views]
    # For multiclass: X = [probs_view1, probs_view2, ...] -> shape [N, num_views * num_classes]
    # For multilabel: X = [view1_label1, view1_label2, ..., view2_label1, ...] -> shape [N, num_views * num_labels]
    
    if task == "binary":
        X_val = np.column_stack([view_predictions[p]["val_probs"] for p in view_planes])
        X_test = np.column_stack([view_predictions[p]["test_probs"] for p in view_planes])
    elif task == "multilabel":
        # For multilabel, concatenate all probability vectors
        # Order: [view1_label1, view1_label2, ..., view1_labelN, view2_label1, ...]
        X_val = np.concatenate([view_predictions[p]["val_probs"] for p in view_planes], axis=1)
        X_test = np.concatenate([view_predictions[p]["test_probs"] for p in view_planes], axis=1)
    else:
        # For multiclass, concatenate probability vectors
        X_val = np.concatenate([view_predictions[p]["val_probs"] for p in view_planes], axis=1)
        X_test = np.concatenate([view_predictions[p]["test_probs"] for p in view_planes], axis=1)
    
    # Labels and sample IDs should be the same across all views - verify consistency
    y_val = view_predictions[view_planes[0]]["val_labels"]
    y_test = view_predictions[view_planes[0]]["test_labels"]
    test_sample_ids = view_predictions[view_planes[0]]["test_sample_ids"]
    
    # Verify consistency across views
    for plane in view_planes[1:]:
        assert np.array_equal(y_val, view_predictions[plane]["val_labels"]), \
            f"Label mismatch between {view_planes[0]} and {plane} views!"
        assert np.array_equal(y_test, view_predictions[plane]["test_labels"]), \
            f"Test label mismatch between {view_planes[0]} and {plane} views!"
        assert test_sample_ids == view_predictions[plane]["test_sample_ids"], \
            f"Sample ID mismatch between {view_planes[0]} and {plane} views!"
    
    logger.info(f"\n{'='*50}")
    logger.info("Step 3: Fitting Logistic Regression Ensemble")
    logger.info(f"{'='*50}")
    logger.info(f"Validation set shape: {X_val.shape}")
    logger.info(f"Test set shape: {X_test.shape}")
    
    # Step 3: Fit logistic regression ensemble on validation predictions
    ensemble = train_logistic_ensemble(X_val, y_val, task)
    val_ensemble_probs = predict_logistic_ensemble_proba(
        ensemble=ensemble,
        X=X_val,
        task=task,
        target_labels=target_labels,
        view_planes=view_planes,
    )
    thresholds = compute_youden_thresholds(
        y_true=y_val,
        y_pred_prob=val_ensemble_probs,
        task=task,
    )
    
    logger.info("\nEnsemble Weights (Logistic Regression Coefficients):")
    if task == "binary":
        for i, plane in enumerate(view_planes):
            logger.info(f"  {plane}: {ensemble.coef_[0][i]:.4f}")
        logger.info(f"  intercept: {ensemble.intercept_[0]:.4f}")
    elif task == "multilabel":
        for label_idx, (label_name, ensemble_label) in enumerate(zip(target_labels, ensemble)):
            logger.info(f"\n  Label: {label_name}")
            for view_idx, plane in enumerate(view_planes):
                logger.info(f"    {plane}: {ensemble_label.coef_[0][view_idx]:.4f}")
            logger.info(f"    intercept: {ensemble_label.intercept_[0]:.4f}")
    else:
        num_classes = len(target_labels)
        num_views = len(view_planes)
        for class_idx, class_name in enumerate(target_labels):
            logger.info(f"\n  Class: {class_name}")
            for view_idx, plane in enumerate(view_planes):
                feature_idx = view_idx * num_classes + class_idx
                logger.info(f"    {plane}: {ensemble.coef_[class_idx][feature_idx]:.4f}")
            logger.info(f"    intercept: {ensemble.intercept_[class_idx]:.4f}")
    
    # Step 4: Evaluate on test set
    logger.info(f"\n{'='*50}")
    logger.info("Step 4: Evaluating Ensemble on Test Set")
    logger.info(f"{'='*50}")
    
    eval_results = evaluate_logistic_ensemble(
        ensemble=ensemble,
        X=X_test,
        y=y_test,
        sample_ids=test_sample_ids,
        task=task,
        target_labels=target_labels,
        view_planes=view_planes,
        output_dir=output_dir,
        class_names=cfg.get("class_names"),
        thresholds=thresholds,
    )
    
    # Save results
    table_dir = os.path.join(output_dir, "table")
    os.makedirs(table_dir, exist_ok=True)
    
    # Save predictions
    eval_results["predictions"].to_csv(os.path.join(table_dir, "predictions.csv"), index=False)
    
    # Save metrics
    if eval_results["metrics"]:
        save_metrics = prepare_saving_metrics(eval_results["metrics"], task)
        save_metrics["ensemble_weights"] = eval_results["ensemble_weights"]
        
        with open(os.path.join(table_dir, "metrics.json"), "w") as f:
            json.dump(save_metrics, f, indent=4)
        
        # Log to wandb
        auroc = save_metrics["AUROC"] if task == "binary" else save_metrics["overall"]["AUROC"]
        auprc = save_metrics["AUPRC"] if task == "binary" else save_metrics["overall"]["AUPRC"]
        wandb.run.summary["test_auroc"] = auroc
        wandb.run.summary["test_auprc"] = auprc
        eval_results["test_auroc"] = auroc
        eval_results["test_auprc"] = auprc
        
        logger.info(f"\nTest AUROC: {auroc:.4f}, Test AUPRC: {auprc:.4f}")
        logger.info(f"Ensemble weights: {eval_results['ensemble_weights']}")
    
    # Save ensemble model(s)
    ensemble_path = os.path.join(output_dir, "ckpt", "logistic_ensemble.pkl")
    os.makedirs(os.path.dirname(ensemble_path), exist_ok=True)
    with open(ensemble_path, "wb") as f:
        pickle.dump(ensemble, f)
    if task == "multilabel":
        logger.info(f"Ensemble models ({len(ensemble)} labels) saved to: {ensemble_path}")
    else:
        logger.info(f"Ensemble model saved to: {ensemble_path}")
    
    return eval_results


def run_multiview_logistic_ensemble_kfold(
    cobra_model: Cobra,
    train_datasets: Dict[str, FeatClassificationDataset],
    test_datasets: Dict[str, FeatClassificationDataset],
    cfg: Dict,
    accelerator: Accelerator,
    output_dir: str,
    view_planes: List[str],
    n_folds: int = 5,
    class_weights: Optional[torch.Tensor] = None,
    train_fraction: float = 1.0,
    subsample_seed: int = 42,
) -> Dict:
    """
    Run K-fold cross-validation for the multi-view logistic regression ensemble.

    For each fold:
      1. Split training data into K-1 train / 1 val (consistent across views).
      2. For each view: train single-view classifier, collect val + test preds.
      3. Stack per-view predictions, fit LogisticRegression on val preds.
      4. Evaluate ensemble on test set, collect AUROC.

    Reports mean +/- std of test AUROC across folds.

    Args:
        cobra_model: Pretrained COBRA model.
        train_datasets: Dict mapping view_plane -> training FeatClassificationDataset.
        test_datasets: Dict mapping view_plane -> test FeatClassificationDataset.
        cfg: Configuration dict.
        accelerator: HuggingFace Accelerator.
        output_dir: Output directory.
        view_planes: List of view plane names (or sequence names).
        n_folds: Number of CV folds.
        class_weights: Optional class weights for loss.

    Returns:
        Dict with per-fold results, mean/std AUROC, and best-fold predictions.
    """
    task = cfg["task"]
    target_labels = cfg["target_labels"]

    if accelerator.is_main_process:
        logger.info(f"\n{'='*50}")
        logger.info(f"{n_folds}-Fold Cross-Validation for Multi-view Logistic Ensemble")
        logger.info(f"View planes: {view_planes}")
        logger.info(f"{'='*50}")

    # Build fold splits from the first view's training dataset (shared sample order across different views)
    ref_dataset = train_datasets[view_planes[0]]
    all_labels = np.array([
        ref_dataset._get_label(sid).numpy() for sid in ref_dataset.sample_ids
    ])

    if task == "multilabel":
        kfold = MultilabelStratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=subsample_seed
        )
        fold_splits = list(kfold.split(
            X=np.arange(len(ref_dataset)), y=all_labels
        ))
    else:
        kfold_splitter = StratifiedKFold(
            n_splits=n_folds, shuffle=True, random_state=subsample_seed
        )
        fold_splits = list(kfold_splitter.split(
            X=np.arange(len(ref_dataset)), y=all_labels
        ))

    fold_aurocs = []
    fold_auprcs = []
    best_fold_auroc = -1.0
    best_fold_eval = None

    for fold_idx, (train_idx, val_idx) in enumerate(fold_splits):
        if accelerator.is_main_process:
            logger.info(f"\n{'='*50}")
            logger.info(f"Fold {fold_idx + 1}/{n_folds}")
            logger.info(f"Train: {len(train_idx)}, Val: {len(val_idx)}")
            logger.info(f"{'='*50}")

        fold_output_dir = os.path.join(output_dir, f"fold_{fold_idx + 1}")
        if accelerator.is_main_process:
            os.makedirs(fold_output_dir, exist_ok=True)

        # Few-shot subsampling applies only to this fold's training indices,
        # shared across channels so per-view predictions remain aligned.
        selected_train_idx, class_weights = prepare_training_indices(
            dataset=ref_dataset,
            candidate_idx=train_idx,
            cfg=cfg,
            accelerator=accelerator,
            train_fraction=train_fraction,
            subsample_seed=subsample_seed + fold_idx,
            context=f"fold {fold_idx + 1}",
        )
        selected_train_idx = selected_train_idx.tolist()

        # Train per-view classifiers and collect predictions for this fold
        view_predictions = {}

        for plane in view_planes:
            if accelerator.is_main_process:
                logger.info(f"\nTraining classifier for: {plane}")

            train_subset = Subset(train_datasets[plane], selected_train_idx)
            val_subset = Subset(train_datasets[plane], val_idx.tolist())

            predictions = collect_predictions_per_view_classifier(
                cobra_model=cobra_model,
                train_subset=train_subset,
                val_subset=val_subset,
                test_dataset=test_datasets[plane],
                view_plane=plane,
                cfg=cfg,
                accelerator=accelerator,
                output_dir=fold_output_dir,
                class_weights=class_weights,
            )
            view_predictions[plane] = predictions

            if accelerator.is_main_process:
                logger.info(
                    f"  {plane} val AUROC: "
                    f"{predictions['best_val_auroc']:.4f}"
                )

        if not accelerator.is_main_process:
            continue

        # Stack predictions across views
        if task == "binary":
            X_val = np.column_stack([
                view_predictions[p]["val_probs"] for p in view_planes
            ])
            X_test = np.column_stack([
                view_predictions[p]["test_probs"] for p in view_planes
            ])
        else:
            X_val = np.concatenate([
                view_predictions[p]["val_probs"] for p in view_planes
            ], axis=1)
            X_test = np.concatenate([
                view_predictions[p]["test_probs"] for p in view_planes
            ], axis=1)

        y_val = view_predictions[view_planes[0]]["val_labels"]
        y_test = view_predictions[view_planes[0]]["test_labels"]
        test_sample_ids = view_predictions[view_planes[0]]["test_sample_ids"]

        # Fit logistic ensemble
        ensemble = train_logistic_ensemble(X_val, y_val, task)
        val_ensemble_probs = predict_logistic_ensemble_proba(
            ensemble=ensemble,
            X=X_val,
            task=task,
            target_labels=target_labels,
            view_planes=view_planes,
        )
        thresholds = compute_youden_thresholds(
            y_true=y_val,
            y_pred_prob=val_ensemble_probs,
            task=task,
        )

        eval_results = evaluate_logistic_ensemble(
            ensemble=ensemble,
            X=X_test,
            y=y_test,
            sample_ids=test_sample_ids,
            task=task,
            target_labels=target_labels,
            view_planes=view_planes,
            output_dir=fold_output_dir,
            class_names=cfg.get("class_names"),
            thresholds=thresholds,
        )

        if eval_results["metrics"]:
            save_metrics = prepare_saving_metrics(
                eval_results["metrics"], task
            )
            save_metrics["ensemble_weights"] = eval_results["ensemble_weights"]
            auroc = (
                save_metrics["AUROC"]
                if task == "binary"
                else save_metrics["overall"]["AUROC"]
            )
            auprc = (
                save_metrics["AUPRC"]
                if task == "binary"
                else save_metrics["overall"]["AUPRC"]
            )

            fold_aurocs.append(auroc)
            fold_auprcs.append(auprc)

            logger.info(
                f"Fold {fold_idx + 1}: AUROC={auroc:.4f}, AUPRC={auprc:.4f}"
            )

            wandb.log({
                f"fold{fold_idx + 1}/test_auroc": auroc,
                f"fold{fold_idx + 1}/test_auprc": auprc,
            })

            # Save fold results
            table_dir = os.path.join(fold_output_dir, "table")
            os.makedirs(table_dir, exist_ok=True)
            eval_results["predictions"].to_csv(
                os.path.join(table_dir, "predictions.csv"), index=False
            )
            with open(os.path.join(table_dir, "metrics.json"), "w") as f:
                json.dump(save_metrics, f, indent=4)

            if auroc > best_fold_auroc:
                best_fold_auroc = auroc
                best_fold_eval = eval_results

    # Aggregate across folds
    if accelerator.is_main_process and fold_aurocs:
        mean_auroc = float(np.mean(fold_aurocs))
        std_auroc = float(np.std(fold_aurocs))
        mean_auprc = float(np.mean(fold_auprcs))
        std_auprc = float(np.std(fold_auprcs))

        logger.info(f"\n{'='*50}")
        logger.info(f"{n_folds}-Fold CV Results (Multi-view Ensemble)")
        logger.info(f"{'='*50}")
        for i, (a, p) in enumerate(zip(fold_aurocs, fold_auprcs)):
            logger.info(f"  Fold {i + 1}: AUROC={a:.4f}, AUPRC={p:.4f}")
        logger.info(f"  Mean AUROC: {mean_auroc:.4f} +/- {std_auroc:.4f}")
        logger.info(f"  Mean AUPRC: {mean_auprc:.4f} +/- {std_auprc:.4f}")

        cv_summary = {
            "n_folds": n_folds,
            "view_planes": view_planes,
            "fold_aurocs": fold_aurocs,
            "fold_auprcs": fold_auprcs,
            "mean_auroc": mean_auroc,
            "std_auroc": std_auroc,
            "mean_auprc": mean_auprc,
            "std_auprc": std_auprc,
        }
        table_dir = os.path.join(output_dir, "table")
        os.makedirs(table_dir, exist_ok=True)
        with open(os.path.join(table_dir, "cv_summary.json"), "w") as f:
            json.dump(cv_summary, f, indent=4)

        wandb.run.summary["cv_mean_auroc"] = mean_auroc
        wandb.run.summary["cv_std_auroc"] = std_auroc
        wandb.run.summary["cv_mean_auprc"] = mean_auprc
        wandb.run.summary["cv_std_auprc"] = std_auprc
        wandb.run.summary["test_auroc"] = mean_auroc
        wandb.run.summary["test_auprc"] = mean_auprc

        # Surface fold-mean scalars so repeat-level aggregation can consume them.
        if best_fold_eval is None:
            best_fold_eval = {}
        best_fold_eval["test_auroc"] = mean_auroc
        best_fold_eval["test_auprc"] = mean_auprc
        best_fold_eval["cv_mean_auroc"] = mean_auroc
        best_fold_eval["cv_std_auroc"] = std_auroc
        best_fold_eval["cv_mean_auprc"] = mean_auprc
        best_fold_eval["cv_std_auprc"] = std_auprc

    return best_fold_eval if best_fold_eval else {}


# =============================================================================
# Attention-based Multi-View Inference
# =============================================================================
def train_multiview_classifier(
    model: MultiViewClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Dict,
    accelerator: Accelerator,
    output_dir: str,
    class_weights: Optional[torch.Tensor] = None,
) -> Dict:
    """
    Train multi-view classifier with attention-based aggregation.
    
    Args:
        model: MultiViewClassifier
        train_loader: Training dataloader
        val_loader: Validation dataloader
        cfg: Configuration dict
        accelerator: HuggingFace Accelerator
        output_dir: Output directory
        class_weights: Optional class weights
    
    Returns:
        Dict with best_model, best_val_auroc
    """
    hyperparams = cfg["hyperparams"]
    task = cfg["task"]
    
    # Setup optimizer (with different learning rates when fine-tuning)
    optimizer = _build_optimizer(model, hyperparams, accelerator)
    
    # Setup scheduler
    max_epochs = hyperparams.get("max_epochs", 100)
    num_training_steps = len(train_loader) * max_epochs
    warmup_ratio = hyperparams.get("warmup_ratio", 0.1)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(warmup_ratio * num_training_steps),
        num_training_steps=num_training_steps,
    )
    
    # Setup loss
    criterion = get_loss_criterion(task, class_weights)
    
    # Prepare with accelerator
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )
    
    # Early stopping
    patience = hyperparams.get("patience", None)
    ckpt_path = os.path.join(output_dir, "checkpoints")
    early_stopping = EarlyStopping(patience=patience, mode="max", ckpt_path=ckpt_path, accelerator=accelerator) if patience else None
    
    if accelerator.is_main_process:
        if patience:
            logger.info(f"Early stopping enabled with patience={patience}")
    
    best_val_auroc = 0.0
    best_state_dict = None
    num_classes = accelerator.unwrap_model(model).num_classes
    
    for epoch in tqdm(range(max_epochs), desc="Training multi-view classifier...", disable=not accelerator.is_main_process):
        # Training
        avg_train_loss, train_metrics = train_per_epoch(model, train_loader, optimizer, scheduler, task, criterion, num_classes, accelerator)
        
        # Validation
        avg_val_loss, val_metrics, val_attentions = eval_per_epoch(model, val_loader, task, criterion, num_classes, accelerator, multi_view=True)
        val_auroc = val_metrics.get("auroc", 0.0)
        
        # Log to wandb
        if accelerator.is_main_process:
            # Log average attention per view
            mean_attention = val_attentions.cpu().mean(dim=0)
            view_planes = accelerator.unwrap_model(model).view_planes
            if epoch % 10 == 0:
                for i, plane in enumerate(view_planes):
                    logger.info(f"Mean attention weight for {plane} view plane: {mean_attention[i].item():.4f}")
            
            wandb.log({
                "lr": scheduler.get_last_lr()[0],
                "train/loss_per_epoch": avg_train_loss,
                "val/loss_per_epoch": avg_val_loss,
                "val/auroc": val_auroc,
                "epoch": epoch + 1,
            })
            for name, value in train_metrics.items():
                wandb.log({f"train/{name}": value})
            for name, value in val_metrics.items():
                wandb.log({f"val/{name}": value})
        
        # Save best model
        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state_dict = accelerator.get_state_dict(model)
            
            if accelerator.is_main_process:
                ckpt_dir = os.path.join(output_dir, "ckpt")
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(best_state_dict, os.path.join(ckpt_dir, "multiview_classifier.pt"))
        
        # Early stopping
        if early_stopping:
            should_stop, best_score = early_stopping.step(
                                            val_score=val_auroc,
                                            model=model,
                                            optimizer=optimizer,
                                            scheduler=scheduler,
                                            epoch=epoch
                                        )
            if should_stop:
                if accelerator.is_main_process:
                    logger.info(f"Early stopping at epoch {epoch} with best validation AUROC {best_score:.4f}.")
                # Load best model on all processes
                best_model = early_stopping.load_best_model(accelerator.unwrap_model(model))
                return {"best_model": best_model, "best_val_auroc": best_score}
    
    if accelerator.is_main_process:
        logger.info(f"Training complete. Best Val AUROC: {best_val_auroc:.4f}")
    
    # Load best weights
    if best_state_dict:
        accelerator.unwrap_model(model).load_state_dict(best_state_dict)
    
    return {"best_model": accelerator.unwrap_model(model), "best_val_auroc": best_val_auroc}


def evaluate_multiview_classifier(
    model: MultiViewClassifier,
    test_loader: DataLoader,
    cfg: Dict,
    accelerator: Accelerator,
    output_dir: str,
    track_emissions: bool = True,
    val_threshold_loader: Optional[DataLoader] = None,
) -> Dict:
    """
    Evaluate multi-view classifier on test set.

    If ``val_threshold_loader`` is provided, validation predictions are used to
    choose decision thresholds for threshold-based test metrics/plots.
    
    Returns predictions, metrics, and attention weights for interpretability.
    """
    task = cfg["task"]
    target_labels = cfg["target_labels"]
    
    # Prepare model and dataloader for distributed evaluation
    if val_threshold_loader is not None:
        model, test_loader, val_threshold_loader = accelerator.prepare(model, test_loader, val_threshold_loader)
    else:
        model, test_loader = accelerator.prepare(model, test_loader)
    model.eval()

    thresholds = None
    if val_threshold_loader is not None:
        val_logits_list = []
        val_labels_list = []
        with torch.no_grad():
            for batch in tqdm(
                val_threshold_loader,
                desc="Collecting validation thresholds...",
                disable=not accelerator.is_main_process,
            ):
                outputs = model(batch["features"], batch["seq_lengths"], batch.get("physical_positions"))
                val_logits_list.append(outputs["logits"].detach())
                val_labels_list.append(batch["labels"].detach())
        val_logits = accelerator.gather_for_metrics(torch.cat(val_logits_list, dim=0))
        val_labels = accelerator.gather_for_metrics(torch.cat(val_labels_list, dim=0))
        if accelerator.is_main_process:
            if task == "binary":
                val_probs = torch.sigmoid(val_logits.squeeze(-1)).cpu().numpy()
            elif task == "multilabel":
                val_probs = torch.sigmoid(val_logits).cpu().numpy()
            else:
                val_probs = F.softmax(val_logits, dim=-1).cpu().numpy()
            thresholds = compute_youden_thresholds(
                y_true=val_labels.cpu().numpy(),
                y_pred_prob=val_probs,
                task=task,
            )
    
    tracker = None
    if track_emissions and accelerator.is_main_process:
        tracker = EmissionsTracker(
            project_name="medslim-inference",
            output_dir=output_dir,
            log_level="warning",
        )
        tracker.start()
    
    all_logits = []
    all_labels = []
    all_sample_ids = []
    all_attention = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating...", disable=not accelerator.is_main_process):
            outputs = model(batch["features"], batch["seq_lengths"], batch.get("physical_positions"))
            all_logits.append(outputs["logits"].detach())
            all_labels.append(batch["labels"].detach())
            all_sample_ids.extend(batch["sample_ids"])
            all_attention.append(outputs["attention_weights"].detach())
    
    if tracker is not None:
        emissions = tracker.stop()
        logger.info(f"Inference emissions: {emissions:.6f} kg CO2eq")
    
    # Gather predictions from all processes
    all_logits = accelerator.gather_for_metrics(torch.cat(all_logits, dim=0))
    all_labels = accelerator.gather_for_metrics(torch.cat(all_labels, dim=0))
    all_attention = accelerator.gather_for_metrics(torch.cat(all_attention, dim=0))
    
    # Only compute metrics on main process
    if not accelerator.is_main_process:
        return {"predictions": None, "metrics": None, "attention_weights": None, "view_planes": None}
    
    # Convert logits to probabilities
    if task == "binary":
        all_probs = torch.sigmoid(all_logits.squeeze(-1)).cpu().numpy()
    elif task == "multilabel":
        all_probs = torch.sigmoid(all_logits).cpu().numpy()
    else:
        all_probs = F.softmax(all_logits, dim=-1).cpu().numpy()
    
    eval_results = _compute_predictions_and_metrics(
        all_probs=all_probs,
        all_true=all_labels.cpu().numpy(),
        sample_ids=all_sample_ids,
        task=task,
        target_labels=target_labels,
        output_dir=output_dir,
        class_names=cfg.get("class_names"),
        thresholds=thresholds,
    )
    
    # Store attention weights for interpretability
    eval_results["attention_weights"] = all_attention.cpu().numpy()
    eval_results["view_planes"] = accelerator.unwrap_model(model).view_planes
    
    return eval_results


def run_multiview_evaluation(
    cobra_model: Cobra,
    train_dataset: MultiViewFeatClassificationDataset,
    test_dataset: MultiViewFeatClassificationDataset,
    cfg: Dict,
    accelerator: Accelerator,
    output_dir: str,
    view_planes: List[str],
    class_weights: Optional[torch.Tensor] = None,
    val_dataset: Optional[MultiViewFeatClassificationDataset] = None,
    train_fraction: float = 1.0,
    subsample_seed: int = 42,
) -> Dict:
    """
    Run complete multi-view evaluation pipeline with attention-based aggregation.
    
    1. Create train/val split (or use dedicated val_dataset if provided)
    2. Optionally subsample the training set to `train_fraction` (few-shot)
    3. Initialize MultiViewClassifier with attention aggregation
    4. Linear probe training
    5. Evaluate on test set
    """
    hyperparams = cfg["hyperparams"]
    linear_hyperparams = hyperparams.get("linear", hyperparams)
    attention_hyperparams = hyperparams.get("attention", {})
    task = cfg["task"]
    
    if accelerator.is_main_process:
        logger.info(f"\n{'='*50}")
        logger.info("Multi-view evaluation with attention aggregation")
        logger.info(f"View planes: {view_planes}")
        logger.info(f"{'='*50}")
    
    # Create train/val split or use dedicated val set
    if val_dataset is not None:
        train_candidate_idx = np.arange(len(train_dataset))
        val_data = val_dataset
        if accelerator.is_main_process:
            logger.info(f"Using dedicated validation set. Training: {len(train_candidate_idx)}, Validation: {len(val_data)}")
    else:
        all_labels = np.array([train_dataset._get_label(sid).numpy() for sid in train_dataset.sample_ids])
        val_split_ratio = hyperparams.get("val_split_ratio", 0.1)
        
        if task == "multilabel":
            n_splits = max(2, int(1.0 / val_split_ratio))
            kfold = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=subsample_seed)
            splits = list(kfold.split(X=np.arange(len(train_dataset)), y=all_labels))
            train_idx, val_idx = splits[0]
        else:
            train_idx, val_idx = train_test_split(
                np.arange(len(train_dataset)),
                test_size=val_split_ratio,
                stratify=all_labels,
                random_state=subsample_seed,
            )
        
        train_candidate_idx = train_idx
        val_data = Subset(train_dataset, val_idx)

    # Few-shot subsampling with subset-aware class weights
    selected_train_idx, class_weights = prepare_training_indices(
        dataset=train_dataset,
        candidate_idx=train_candidate_idx,
        cfg=cfg,
        accelerator=accelerator,
        train_fraction=train_fraction,
        subsample_seed=subsample_seed,
        context="multiview-attention",
    )
    train_data = Subset(train_dataset, selected_train_idx.tolist())
    if accelerator.is_main_process:
        logger.info(f"Split training set. Training: {len(train_data)}, Validation: {len(val_data)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=hyperparams["batch_size"], shuffle=True,
                              collate_fn=multiview_classifier_collate_fn, num_workers=hyperparams.get("num_workers", 0))
    val_loader = DataLoader(val_data, batch_size=hyperparams["batch_size"], shuffle=False,
                            collate_fn=multiview_classifier_collate_fn, num_workers=hyperparams.get("num_workers", 0))
    test_loader = DataLoader(test_dataset, batch_size=hyperparams["batch_size"], shuffle=False,
                             collate_fn=multiview_classifier_collate_fn, num_workers=hyperparams.get("num_workers", 0))
    
    # Get dimensions
    num_classes = get_num_classes(task, cfg["target_labels"], train_dataset)
    input_dim = cobra_model.output_dim
    
    if accelerator.is_main_process:
        logger.info(f"Num classes: {num_classes}, COBRA output dim: {input_dim}")
    
    # Initialize model
    freeze_cobra = cfg.get("freeze_cobra", True)
    model = MultiViewClassifier(
        cobra_model=cobra_model,
        view_planes=view_planes,
        input_dim=input_dim,
        num_classes=num_classes,
        classifier_hidden_dim=linear_hyperparams.get("hidden_dim", 512),
        classifier_dropout=linear_hyperparams.get("dropout", 0.5),
        attention_hidden_dim=attention_hyperparams.get("hidden_dim", 128),
        attention_dropout=attention_hyperparams.get("dropout", 0.1),
        freeze_cobra=freeze_cobra,
        fm_ids=cfg.get("eval_fm_ids"),
    )
    
    if accelerator.is_main_process:
        mode = "Linear Probing" if freeze_cobra else "Fine-tuning"
        logger.info(f"Training mode: {mode}")
    _log_trainable_params(model, accelerator)
    
    # Train
    training_results = train_multiview_classifier(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        accelerator=accelerator,
        output_dir=output_dir,
        class_weights=class_weights,
    )
    
    best_model = training_results["best_model"]
    
    if accelerator.is_main_process:
        logger.info(f"\n{'='*50}")
        logger.info("Evaluating on test set")
        logger.info(f"{'='*50}")
    
    # Evaluate
    eval_results = evaluate_multiview_classifier(
        model=best_model,
        test_loader=test_loader,
        cfg=cfg,
        accelerator=accelerator,
        output_dir=output_dir,
        val_threshold_loader=val_loader,
    )
    
    # Wait for all processes before saving
    accelerator.wait_for_everyone()
    
    # Save results
    if accelerator.is_main_process:
        table_dir = os.path.join(output_dir, "table")
        os.makedirs(table_dir, exist_ok=True)
        
        # Save predictions with attention weights
        eval_results["predictions"].to_csv(os.path.join(table_dir, "predictions.csv"), index=False)
        
        # Save metrics
        if eval_results["metrics"]:
            save_metrics = prepare_saving_metrics(eval_results["metrics"], task)
            
            # Add mean attention weights to metrics
            mean_attention = eval_results["attention_weights"].mean(axis=0)
            save_metrics["attention_weights"] = {
                plane: mean_attention[i].item() for i, plane in enumerate(view_planes)
            }
            
            with open(os.path.join(table_dir, "metrics.json"), "w") as f:
                json.dump(save_metrics, f, indent=4)
            
            # Log to wandb
            auroc = save_metrics["AUROC"] if task == "binary" else save_metrics["overall"]["AUROC"]
            auprc = save_metrics["AUPRC"] if task == "binary" else save_metrics["overall"]["AUPRC"]
            wandb.run.summary["test_auroc"] = auroc
            wandb.run.summary["test_auprc"] = auprc
            eval_results["test_auroc"] = auroc
            eval_results["test_auprc"] = auprc
            
            for plane in view_planes:
                wandb.run.summary[f"test_attention_{plane}"] = save_metrics["attention_weights"][plane]
            
            logger.info(f"Test AUROC: {auroc:.4f}, Test AUPRC: {auprc:.4f}")
            logger.info(f"Attention weights: {save_metrics['attention_weights']}")
    
    return eval_results


# =============================================================================
# Main Entry Point
# =============================================================================

def main(args):
    """Main function for linear probing evaluation."""
    # Initialize accelerator
    accelerator = Accelerator()

    # Load config
    with open(args.linear_classifier_config, "r") as f:
        cfg = yaml.safe_load(f)

    dataset_name = cfg.get("feat_dataset", {}).get("dataset_name")
    if dataset_name is None:
        raise ValueError("Missing required field `dataset_name` in `linear_classifier.yml`.")
    ds_meta = get_dataset_metadata(dataset_name)
    if "task" not in ds_meta:
        raise ValueError("Missing required field `task` in `eval_datasets.yaml` for dataset `dataset_name`.")
    if "target_labels" not in ds_meta:
        raise ValueError("Missing required field `target_labels` in `eval_datasets.yaml` for dataset `dataset_name`.")
    cfg["task"] = ds_meta["task"]
    cfg["target_labels"] = ds_meta["target_labels"]
    if cfg["task"] == "multiclass":
        if "multiclass_label_maps" not in ds_meta:
            raise ValueError("Missing required field `multiclass_label_maps` in `eval_datasets.yaml` for dataset `dataset_name`.")
        cfg["class_names"] = build_multiclass_label_names(
            cfg["target_labels"][0], ds_meta["multiclass_label_maps"]
        )

    # Get annotation paths from annotations_dir
    annotations_dir = cfg.get("annotations_dir")
    if not annotations_dir:
        raise ValueError("Missing required field `annotations_dir` in linear_classifier.yml")
    annot_files = get_annotation_paths_by_split(
        annotations_dir, cfg["task"],
        splits=["train", "test"],
        optional_splits=["val"],
    )
    cfg["train_annots"] = str(annot_files["train"])
    cfg["test_annots"] = str(annot_files["test"])
    if "val" in annot_files:
        cfg["val_annots"] = str(annot_files["val"])

    # CLI `--fine-tune` flag overrides `freeze_cobra` in config
    if args.fine_tune:
        cfg["freeze_cobra"] = not args.fine_tune

    # Few-shot / repeated-evaluation configuration.
    # CLI overrides config under cfg["few_shot"], which itself overrides defaults.
    few_shot_cfg = cfg.get("few_shot", {}) or {}
    train_fraction = (
        args.train_fraction
        if args.train_fraction is not None
        else few_shot_cfg.get("train_fraction", 1.0)
    )
    num_repeats = (
        args.num_repeats
        if args.num_repeats is not None
        else few_shot_cfg.get("num_repeats", 1)
    )
    if not 0.0 < train_fraction <= 1.0:
        raise ValueError(f"--train-fraction must be in (0, 1], got {train_fraction}")
    if num_repeats < 1:
        raise ValueError(f"--num-repeats must be >= 1, got {num_repeats}")
    cfg["weighted_loss"] = args.weighted_loss
    cfg["train_fraction"] = train_fraction
    cfg["num_repeats"] = num_repeats

    # Generate one random seed per repeat. Seeds are generated at runtime to avoid cherry-picking favorable seeds.
    seed_sequence = (
        np.random.SeedSequence(args.seed)
        if args.seed is not None
        else np.random.SeedSequence()
    )
    repeat_seeds = [int(s) for s in seed_sequence.generate_state(num_repeats)]
    cfg["repeat_seeds"] = repeat_seeds

    # Get view plane(s) and model names from config
    planes = cfg["feat_dataset"]["plane"]
    if args.feat_dir:
        cfg["feat_dataset"]["feat_dir"] = args.feat_dir

    if args.fm_model_names:
        model_names = args.fm_model_names.split()  
    else:
        model_names = cfg["feat_dataset"]["model_name"]
    if isinstance(model_names, str):
        model_names = [model_names]
    
    logger.info(f"FM choices: {model_names}")

    # Build the channel map: list of (channel_name, feat_dir, plane).
    # Each channel becomes an independent entry in the logistic ensemble.
    #   - Single-sequence: channels = planes  (e.g. ["sagittal", "coronal"])
    #   - Multi-sequence:  channels = seq x plane  (e.g. ["DESS_E1/sagittal", "DESS_E2/sagittal"])
    sequences = cfg["feat_dataset"].get("sequences")
    if sequences:
        channel_map = [
            (f"{seq_name}/{plane}", seq_feat_dir, plane)
            for seq_name, seq_feat_dir in sequences.items()
            for plane in planes
        ]
    else:
        single_feat_dir = cfg["feat_dataset"]["feat_dir"]
        channel_map = [
            (plane, single_feat_dir, plane)
            for plane in planes
        ]

    view_planes = [ch[0] for ch in channel_map]
    use_multiview = len(view_planes) > 1

    # Create output directory
    target_labels = "_".join(cfg["target_labels"])
    fewshot_tag = ""
    if train_fraction < 1.0:
        fewshot_tag += f"_frac{train_fraction:g}"
    if num_repeats > 1:
        fewshot_tag += f"_repeats{num_repeats}"
    if use_multiview:
        planes_str = "_".join(view_planes)
        output_dir = os.path.join(cfg["output_dir"], f"{target_labels}_multiview_{planes_str}{fewshot_tag}_{CURR_TIME}_{JOB_ID}")
    else:
        output_dir = os.path.join(cfg["output_dir"], f"{target_labels}_{view_planes[0]}{fewshot_tag}_{CURR_TIME}_{JOB_ID}")
    
    checkpoint_path = args.checkpoint_path if args.checkpoint_path else cfg.get("checkpoint_path")
    pretrain_config_path = (
        args.pretrain_config_path
        or cfg.get("pretrain_config_path")
        or (Path(checkpoint_path).parent / "config.yaml" if checkpoint_path else None)
    )
    if pretrain_config_path is None:
        raise ValueError(
            "A pretraining config is required. Provide --pretrain-config-path "
            "or set pretrain_config_path/checkpoint_path in the config."
        )
    with open(pretrain_config_path, "r") as f:
        pretrain_cfg = yaml.safe_load(f)
    cobra_cfg = pretrain_cfg["model"]["cobra"]
    checkpoint_slice_pooling = None
    pretrain_state = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        pretrain_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_slice_pooling = pretrain_state.get("pooling")
        cobra_cfg["physical_pe"] = bool(
            pretrain_state.get("physical_pe", cobra_cfg.get("physical_pe", False))
        )
        cobra_cfg["regional_tokens"] = int(
            pretrain_state.get("regional_tokens", cobra_cfg.get("regional_tokens", 0))
        )
    fm_pooling = (
        args.fm_pooling
        or (pretrain_state.get("fm_pooling") if pretrain_state is not None else None)
        or cobra_cfg.get("fm_pooling", "avg_pool")
    )
    if fm_pooling not in ("avg_pool", "router"):
        raise ValueError(
            f"Invalid fm_pooling '{fm_pooling}'. Must be 'avg_pool' or 'router'."
        )
    fm_id_order = (
        pretrain_state.get("fm_id_order") if pretrain_state is not None else None
    ) or pretrain_cfg.get("feat_dataset", {}).get("model_name")
    eval_fm_ids = resolve_eval_fm_ids(
        fm_pooling, model_names, pretrain_cfg, pretrain_state
    )
    cfg["cobra_config"] = cobra_cfg
    cfg["physical_pe"] = cobra_cfg.get("physical_pe", False)
    cfg["regional_tokens"] = cobra_cfg.get("regional_tokens", 0)
    cfg["fm_id_order"] = fm_id_order
    cfg["eval_fm_ids"] = eval_fm_ids
    if pretrain_state is not None:
        cfg["num_fms"] = pretrain_state.get("num_fms")
        cfg["router_mode"] = pretrain_state.get("router_mode", cobra_cfg.get("router_mode", "soft"))
        cfg["router_top_k"] = pretrain_state.get("router_top_k", cobra_cfg.get("router_top_k"))
        cfg["router_temperature"] = pretrain_state.get("router_temperature", cobra_cfg.get("router_temperature", 1.0))
        cfg["router_learnable_temperature"] = pretrain_state.get(
            "router_learnable_temperature",
            cobra_cfg.get("router_learnable_temperature", False),
        )
    if accelerator.is_main_process and eval_fm_ids is not None:
        logger.info(f"Router eval FM IDs for {model_names}: {eval_fm_ids}")
    resolved_pooling_target = _resolve_pooling_target(
        mode="inference",
        pooling_target=args.pooling_target,
        regional_tokens=cobra_cfg.get("regional_tokens", 0),
        slice_pooling=args.slice_pooling or checkpoint_slice_pooling or cobra_cfg.get("pooling", "abmil"),
    )
    if fm_pooling == "router" and resolved_pooling_target == "raw":
        raise ValueError(
            f"pooling_target='raw' bypasses FM fusion and is not valid with fm_pooling='{fm_pooling}'. "
            "Use pooling_target='post_embed' or 'post_encoder'."
        )
    cfg["fm_pooling"] = fm_pooling
    cfg["pooling_target"] = resolved_pooling_target
    cfg["sequence_encoder"] = (
        args.sequence_encoder
        or (pretrain_state.get("sequence_encoder") if pretrain_state is not None else None)
        or "mamba2"
    )
    cfg["slice_pooling"] = args.slice_pooling or checkpoint_slice_pooling or cobra_cfg.get("pooling", "abmil")

    # Determine raw FM output dimension for 'raw' pooling target.
    # Tiled caches are [B, num_slices, num_tiled_regions, input_embed_dim], so
    # raw ABMIL pooling over [B, num_slices, input_embed_dim] is not defined.
    raw_output_dim = None
    if resolved_pooling_target == "raw":
        fm_configs = {m["name"]: m for m in pretrain_cfg["model"]["slice_encoder_models"]}
        raw_output_dim = fm_configs[model_names[0]]["embed_dim"]
    cfg["raw_output_dim"] = raw_output_dim

    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "config.yml"), "w") as f:
            yaml.dump(cfg, f)

    # Initialize wandb
    if accelerator.is_main_process:
        mode_str = "multiview_ensemble" if use_multiview else "single_view"
        wandb.init(
            project="medslim-linear-probing",
            name=f"{cfg['feat_dataset']['dataset_name']}_{target_labels}_{mode_str}",
            config=cfg,
            dir=output_dir,
        )
    if checkpoint_path is None:
        raise ValueError(
            "checkpoint_path is required for downstream SSL evaluation. "
            "Set it in linear_classifier.yml or pass --checkpoint-path."
        )
    if accelerator.is_main_process:
        logger.info("Loading pretrained COBRA model...")
        logger.info(f"Checkpoint path: {checkpoint_path}")
    cobra_model = load_pretrained_cobra(
        checkpoint_path=checkpoint_path,
        accelerator=accelerator,
        model_config=cobra_cfg,
        encoder_type=cfg["encoder_type"],
        fm_pooling=fm_pooling,
        sequence_encoder=args.sequence_encoder,
        slice_pooling=args.slice_pooling,
        pooling_target=resolved_pooling_target,
        raw_output_dim=raw_output_dim,
    )
    cobra_model = cobra_model.to(accelerator.device)
    cobra_model.eval()

    # Run evaluation
    has_val_annots = "val_annots" in cfg and cfg["val_annots"]

    train_datasets = test_datasets = val_datasets = None
    train_dataset = test_dataset = val_dataset = None

    if use_multiview:
        # Multi-view / multi-sequence logistic regression ensemble.
        # Each entry in channel_map is (channel_name, feat_dir, plane).
        if accelerator.is_main_process:
            logger.info("Using multi-channel logistic regression ensemble")
            logger.info(f"Channels: {view_planes}")
        
        # Create datasets for every channel
        train_datasets = {}
        test_datasets = {}
        val_datasets = {} if has_val_annots else None
        
        for channel_name, feat_dir, plane in channel_map:
            train_datasets[channel_name] = FeatClassificationDataset(
                feat_dir=feat_dir,
                slice_encoder_models=model_names,
                view_plane=plane,
                split="train",
                annotations_path=cfg["train_annots"],
                task=cfg["task"],
                target_columns=cfg["target_labels"],
                cache_in_memory=True,
            )
            
            test_datasets[channel_name] = FeatClassificationDataset(
                feat_dir=feat_dir,
                slice_encoder_models=model_names,
                view_plane=plane,
                split="test",
                annotations_path=cfg["test_annots"],
                task=cfg["task"],
                target_columns=cfg["target_labels"],
                cache_in_memory=True,
            )
            
            if has_val_annots:
                val_datasets[channel_name] = FeatClassificationDataset(
                    feat_dir=feat_dir,
                    slice_encoder_models=model_names,
                    view_plane=plane,
                    split="val",
                    annotations_path=cfg["val_annots"],
                    task=cfg["task"],
                    target_columns=cfg["target_labels"],
                    cache_in_memory=True,
                )
        
    else:
        # Single-view channel
        channel_name, feat_dir, plane = channel_map[0]
        
        train_dataset = FeatClassificationDataset(
            feat_dir=feat_dir,
            slice_encoder_models=model_names,
            view_plane=plane,
            split="train",
            annotations_path=cfg["train_annots"],
            task=cfg["task"],
            target_columns=cfg["target_labels"],
            cache_in_memory=True,
        )
        
        test_dataset = FeatClassificationDataset(
            feat_dir=feat_dir,
            slice_encoder_models=model_names,
            view_plane=plane,
            split="test",
            annotations_path=cfg["test_annots"],
            task=cfg["task"],
            target_columns=cfg["target_labels"],
            cache_in_memory=True,
        )
        
        val_dataset = None
        if has_val_annots:
            val_dataset = FeatClassificationDataset(
                feat_dir=feat_dir,
                slice_encoder_models=model_names,
                view_plane=plane,
                split="val",
                annotations_path=cfg["val_annots"],
                task=cfg["task"],
                target_columns=cfg["target_labels"],
                cache_in_memory=True,
            )
        
        if accelerator.is_main_process:
            logger.info(f"Using single-channel inference: {channel_name}")
            logger.info(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
            if val_dataset is not None:
                logger.info(f"Val samples (dedicated): {len(val_dataset)}")

    # K-fold is only applicable when there is no dedicated validation split.
    # When a dedicated val split exists, uncertainty is estimated via repeated
    # random runs (num_repeats), not by reinterpreting --n-folds.
    has_dedicated_val = (val_datasets is not None) if use_multiview else (val_dataset is not None)
    use_kfold = args.n_folds > 1 and not has_dedicated_val
    if args.n_folds > 1 and has_dedicated_val and accelerator.is_main_process:
        logger.warning(
            f"--n-folds={args.n_folds} ignored because a dedicated validation set is "
            f"available; using num_repeats={num_repeats} for uncertainty instead."
        )

    def _run_single_repeat(repeat_output_dir: str, seed: int) -> Dict:
        """Run one full evaluation for a given output dir and seed."""
        if use_multiview:
            if use_kfold:
                return run_multiview_logistic_ensemble_kfold(
                    cobra_model=cobra_model,
                    train_datasets=train_datasets,
                    test_datasets=test_datasets,
                    cfg=cfg,
                    accelerator=accelerator,
                    output_dir=repeat_output_dir,
                    view_planes=view_planes,
                    n_folds=args.n_folds,
                    train_fraction=train_fraction,
                    subsample_seed=seed,
                )
            return run_multiview_logistic_ensemble(
                cobra_model=cobra_model,
                train_datasets=train_datasets,
                test_datasets=test_datasets,
                cfg=cfg,
                accelerator=accelerator,
                output_dir=repeat_output_dir,
                view_planes=view_planes,
                val_datasets=val_datasets,
                train_fraction=train_fraction,
                subsample_seed=seed,
            )
        if use_kfold:
            return run_single_view_kfold_evaluation(
                cobra_model=cobra_model,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                cfg=cfg,
                accelerator=accelerator,
                output_dir=repeat_output_dir,
                n_folds=args.n_folds,
                train_fraction=train_fraction,
                subsample_seed=seed,
            )
        return run_single_view_evaluation(
            cobra_model=cobra_model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            cfg=cfg,
            accelerator=accelerator,
            output_dir=repeat_output_dir,
            val_dataset=val_dataset,
            train_fraction=train_fraction,
            subsample_seed=seed,
        )

    # Repeated random evaluation when num_repeats > 1
    repeat_metrics: List[Dict] = []
    for repeat_idx, seed in enumerate(repeat_seeds):
        set_global_seed(seed)
        repeat_output_dir = (
            os.path.join(output_dir, f"repeat_{repeat_idx + 1}")
            if num_repeats > 1 else output_dir
        )
        if accelerator.is_main_process:
            os.makedirs(repeat_output_dir, exist_ok=True)
            logger.info(f"\n{'#'*60}")
            logger.info(f"Repeat {repeat_idx + 1}/{num_repeats} (seed={seed})")
            logger.info(f"{'#'*60}")

        eval_results = _run_single_repeat(repeat_output_dir, seed)

        if accelerator.is_main_process:
            repeat_metrics.append({
                "repeat": repeat_idx + 1,
                "seed": seed,
                "test_auroc": eval_results.get("test_auroc") if eval_results else None,
                "test_auprc": eval_results.get("test_auprc") if eval_results else None,
            })

    # Aggregate test metrics across repeats
    if accelerator.is_main_process:
        summary = aggregate_repeat_metrics(repeat_metrics)
        repeat_summary = {
            "num_repeats": num_repeats,
            "train_fraction": train_fraction,
            "n_folds": args.n_folds if use_kfold else 1,
            "repeat_seeds": repeat_seeds,
            "per_repeat": repeat_metrics,
            **summary,
        }
        table_dir = os.path.join(output_dir, "table")
        os.makedirs(table_dir, exist_ok=True)
        with open(os.path.join(table_dir, "repeat_summary.json"), "w") as f:
            json.dump(repeat_summary, f, indent=4)

        wandb.run.summary["train_fraction"] = train_fraction
        wandb.run.summary["num_repeats"] = num_repeats
        for key, value in summary.items():
            wandb.run.summary[key] = value
        if "mean_auroc" in summary:
            wandb.run.summary["test_auroc"] = summary["mean_auroc"]
            wandb.run.summary["test_auprc"] = summary["mean_auprc"]
            logger.info(
                f"Repeated evaluation ({num_repeats} repeat(s), train_fraction={train_fraction}): "
                f"AUROC {summary['mean_auroc']:.4f} +/- {summary['std_auroc']:.4f}, "
                f"AUPRC {summary['mean_auprc']:.4f} +/- {summary['std_auprc']:.4f}"
            )

    if accelerator.is_main_process:
        logger.info(f"\n{'='*50}")
        logger.info("Evaluation Complete!")
        logger.info(f"Results saved to: {output_dir}")
        logger.info(f"{'='*50}")
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear Probing Evaluation for MedSliM")
    parser.add_argument(
        "--linear-classifier-config",
        type=str,
        required=True,
        help="Path to the linear classifier config file"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Path to pretrained COBRA checkpoint. Overrides config file if provided."
    )
    parser.add_argument(
        "--pretrain-config-path",
        type=str,
        default=None,
        help=(
            "Path to the MedSliM pretraining config. If specified, overrides the config file."
        )
    )
    parser.add_argument(
        "--feat-dir",
        type=str,
        default=None,
        help="Override feat_dataset.feat_dir (e.g. .../crop_tiled_2x2 for tiled checkpoints).",
    )
    parser.add_argument(
        "--fm-model-names",
        type=str,
        default=None,
        help="Foundation model names for slice feature extraction. If not provided, uses model names from config file."
    )
    parser.add_argument(
        "--weighted-loss",
        action="store_true",
        help="Use weighted loss for training"
    )
    parser.add_argument(
        "--sequence-encoder",
        type=str,
        choices=["mamba2", "transformer"],
        default=None,
    )
    parser.add_argument(
        "--fine-tune",
        action="store_true",
        help="Unfreeze COBRA backbone. Overrides freeze_cobra in config."
    )
    parser.add_argument(
        "--fm-pooling",
        type=str,
        choices=["avg_pool", "router"],
        default=None,
        help=(
            "Foundation model pooling method: 'avg_pool' or 'router'. "
            "If not specified, uses the checkpoint's saved fm_pooling when available; "
            "router requires matching pretrained weights for SSL evaluation."
        ),
    )
    parser.add_argument(
        "--slice-pooling",
        type=str,
        choices=["abmil", "cls"],
        default=None,
    )
    parser.add_argument(
        "--pooling-target",
        type=str,
        choices=["post_encoder", "post_embed", "raw"],
        default=None,
        help="Which representation level ABMIL attention weights aggregate: "
             "'post_encoder': after sequence encoder (Mamba-2/Transformer), "
             "'post_embed': after Embed MLP; tiled multi-crop CLS tokens are flattened before pooling. "
             "'raw': original FM embeddings; tiled tokens are flattened before pooling. "
             "If omitted, Cobra resolves to 'raw' for global-only caches and 'post_embed' for tiled multi-crop CLS caches."
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=1,
        help="Number of cross-validation folds. "
             "When > 1, runs K-fold CV and reports mean +/- std AUROC. "
             "Ignored when a dedicated validation set is available. (default: 1)"
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=None,
        help="Few-shot label budget: fraction of the training set used to train "
             "the classifier head, in (0, 1]. Validation/test are never subsampled. "
             "Overrides few_shot.train_fraction in the config (default: 1.0)."
    )
    parser.add_argument(
        "--num-repeats",
        type=int,
        default=None,
        help="Number of repeated random runs for uncertainty estimation. Each "
             "repeat uses a freshly generated seed and (when > 1) its own output "
             "subdirectory; test metrics are aggregated as mean +/- std. Overrides "
             "few_shot.num_repeats in the config (default: 1)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional base seed for generating per-repeat seeds. Use only for "
             "debugging or exact reruns; by default seeds are drawn from system "
             "entropy and recorded in the run outputs."
    )
    args = parser.parse_args()
    main(args)
