"""
Linear Probing Evaluation for pretrained MedSliM SSL Model.

- Single-view classification: Train linear classifier head
- Multi-view classification: 
  1. Train independent single-view classifiers for each view plane
  2. Collect predictions from each view
  3. Fit logistic regression on stacked predictions
  4. Evaluate ensemble on test set
"""
import os
import argparse
import warnings
import yaml
import json
import logging
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from datetime import datetime
from accelerate import Accelerator
from typing import Dict, List, Optional, Tuple
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from transformers import get_cosine_schedule_with_warmup

from med_slim.model.sequence_encoder.cobra import Cobra
from med_slim.model.attention_pooling.abmil import BatchedABMIL
from med_slim.data.feat_dataset import (
    FeatClassificationDataset, 
    MultiViewFeatClassificationDataset,
    linear_classifier_collate_fn,
    multiview_classifier_collate_fn,
)
from med_slim.eval.load_cobra import load_pretrained_cobra
from med_slim.utils.callbacks.early_stopping import EarlyStopping
from med_slim.utils.metrics.linear import get_loss_criterion, get_eval_metrics, get_num_classes, compute_class_weights_for_weighted_loss
from med_slim.utils.viz.linear import compute_and_visualize_metrics
from med_slim.logging.setup import init_logging

init_logging()
logger = logging.getLogger(__name__)

CURR_TIME = datetime.now().strftime("%Y-%m-%d-%H:%M")


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
    ):
        super().__init__()
        self.num_classes = num_classes
        self.freeze_cobra = freeze_cobra
        
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
    
    def forward(self, features: List[torch.Tensor], seq_lengths: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: List of K tensors, each [B, max_seq_len, encoder_embed_dim]
                      COBRA inference mode ensembles across these encoders
            seq_lengths: Sequence lengths [B]
        
        Returns:
            Dict with logits and embedding
        """
        # Keep COBRA in eval mode when frozen to disable dropout
        if self.freeze_cobra:
            self.cobra.eval()
        
        # Cast features to model dtype
        features = [f.to(dtype=next(self.cobra.parameters()).dtype) for f in features]
        embedding = self.cobra(features, seq_lengths=seq_lengths)  # [B, embed_dim]
        
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
    ):
        super().__init__()
        self.view_planes = view_planes
        self.num_views = len(view_planes)
        self.num_classes = num_classes
        self.freeze_cobra = freeze_cobra
        
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
    
    def forward(self, features: Dict[str, List[torch.Tensor]], seq_lengths: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: {view_plane: List of K tensors [B, max_seq_len, encoder_embed_dim]}
            seq_lengths: {view_plane: [B]}
        
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
            emb = self.cobra(feats, seq_lengths=seq_length)  # [B, embed_dim]
            
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
) -> Dict:
    """Convert probabilities to prediction labels, compute metrics, and create results DataFrame."""
    if task == "binary":
        all_probs = all_probs.squeeze(-1) if all_probs.ndim > 1 else all_probs
        all_preds = (all_probs >= 0.5).astype(int)
    elif task == "multilabel":
        all_preds = (all_probs >= 0.5).astype(int)
    else:
        all_preds = all_probs.argmax(axis=-1)
    
    # Compute metrics 
    viz_metrics = None
    if output_dir is not None:
        fig_dir = os.path.join(output_dir, "fig")
        viz_labels = target_labels[:1] if task == "binary" else target_labels
        
        viz_metrics = compute_and_visualize_metrics(
            y_true=all_true,
            y_pred_prob=all_probs,
            task=task,
            class_labels=viz_labels,
            output_dir=fig_dir,
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
            outputs = model(batch["features"], batch["seq_lengths"])
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
            outputs = model(batch["features"], batch["seq_lengths"])
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
    
    # Setup optimizer (only train unfrozen parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(hyperparams.get("lr", 1e-4)),
        weight_decay=1e-4
    )
    
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
) -> Dict:
    """Evaluate single-view classifier on test set."""
    task = cfg["task"]
    target_labels = cfg["target_labels"]
    
    # Prepare model and dataloader for distributed evaluation
    model, test_loader = accelerator.prepare(model, test_loader)
    model.eval()
    
    all_logits = []
    all_labels = []
    all_sample_ids = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating...", disable=not accelerator.is_main_process):
            outputs = model(batch["features"], batch["seq_lengths"])
            all_logits.append(outputs["logits"].detach())
            all_labels.append(batch["labels"].detach())
            all_sample_ids.extend(batch["sample_ids"])
    
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
) -> Dict:
    """
    Run complete single-view evaluation pipeline.
    
    1. Create train/val split
    2. Initialize SingleViewClassifier (COBRA + classifier head)
    3. Linear probe training
    4. Evaluate on test set
    """
    hyperparams = cfg["hyperparams"]
    linear_hyperparams = hyperparams.get("linear", hyperparams)
    task = cfg["task"]
    
    if accelerator.is_main_process:
        logger.info(f"\n{'='*50}")
        logger.info("Single-view evaluation")
        logger.info(f"{'='*50}")
    
    # Create train/val split
    all_labels = np.array([train_dataset._get_label(sid).numpy() for sid in train_dataset.sample_ids])
    val_split_ratio = hyperparams.get("val_split_ratio", 0.1)
    
    if task == "multilabel":
        n_splits = max(2, int(1.0 / val_split_ratio))
        kfold = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True)
        splits = list(kfold.split(X=np.arange(len(train_dataset)), y=all_labels))
        train_idx, val_idx = splits[0]
    else:
        train_idx, val_idx = train_test_split(
            np.arange(len(train_dataset)),
            test_size=val_split_ratio,
            stratify=all_labels,
            shuffle=True
        )
    
    if accelerator.is_main_process:
        logger.info(f"Training samples: {len(train_idx)}, Validation samples: {len(val_idx)}")
    
    # Create subsets and dataloaders
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(train_dataset, val_idx)
    
    train_loader = DataLoader(train_subset, batch_size=hyperparams["batch_size"], shuffle=True, 
                              collate_fn=linear_classifier_collate_fn, num_workers=hyperparams.get("num_workers", 0))
    val_loader = DataLoader(val_subset, batch_size=hyperparams["batch_size"], shuffle=False,
                            collate_fn=linear_classifier_collate_fn, num_workers=hyperparams.get("num_workers", 0))
    test_loader = DataLoader(test_dataset, batch_size=hyperparams["batch_size"], shuffle=False,
                             collate_fn=linear_classifier_collate_fn, num_workers=hyperparams.get("num_workers", 0))
    
    # Get dimensions
    num_classes = get_num_classes(task, cfg["target_labels"], train_dataset)
    input_dim = cobra_model.embed_dim
    
    if accelerator.is_main_process:
        logger.info(f"Num classes: {num_classes}, COBRA output dim: {input_dim}")
    
    # Initialize model
    model = SingleViewClassifier(
        cobra_model=cobra_model,
        input_dim=input_dim,
        num_classes=num_classes,
        classifier_hidden_dim=linear_hyperparams.get("hidden_dim", 512),
        classifier_dropout=linear_hyperparams.get("dropout", 0.5),
        freeze_cobra=True,
    )
    
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
    
    return eval_results


# =============================================================================
# Multi-View Logistic Regression Ensemble
# =============================================================================
def collect_predictions_per_view_classifier(
    cobra_model: Cobra,
    train_subset: Subset,
    val_subset: Subset,
    test_dataset: FeatClassificationDataset,
    view_plane: str,
    cfg: Dict,
    accelerator: Accelerator,
    output_dir: str,
    class_weights: Optional[torch.Tensor] = None,
) -> Dict:
    """
    Train a single-view classifier to get per-view predictions on validation and test sets for logistic regression ensemble.
    
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
    num_classes = get_num_classes(task, cfg["target_labels"], train_subset.dataset)
    input_dim = cobra_model.embed_dim
    
    # Initialize model
    model = SingleViewClassifier(
        cobra_model=cobra_model,
        input_dim=input_dim,
        num_classes=num_classes,
        classifier_hidden_dim=linear_hyperparams.get("hidden_dim", 512),
        classifier_dropout=linear_hyperparams.get("dropout", 0.5),
        freeze_cobra=True,
    )
    
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
            outputs = best_model(batch["features"], batch["seq_lengths"])
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
            outputs = best_model(batch["features"], batch["seq_lengths"])
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


def evaluate_logistic_ensemble(
    ensemble: LogisticRegression or List[LogisticRegression],
    X: np.ndarray,
    y: np.ndarray,
    sample_ids: List,
    task: str,
    target_labels: List[str],
    view_planes: List[str],
    output_dir: Optional[str] = None,
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
    
    Returns:
        Dict with predictions DataFrame, metrics, and ensemble weights
    """
    # Get predictions
    if task == "binary":
        y_prob = ensemble.predict_proba(X)[:, 1]
    elif task == "multilabel":
        # For multilabel, get predictions from each label's ensemble
        num_labels = len(target_labels)
        num_views = len(view_planes)
        y_prob_list = []
        
        for label_idx, ensemble_label in enumerate(ensemble):
            # Extract probabilities for this label from all views
            X_label = np.column_stack([
                X[:, view_idx * num_labels + label_idx] 
                for view_idx in range(num_views)
            ])
            # Get probability of positive class for this label
            y_prob_label = ensemble_label.predict_proba(X_label)[:, 1]
            y_prob_list.append(y_prob_label)
        
        # Stack: [N, num_labels]
        y_prob = np.column_stack(y_prob_list)
    else:
        # Multiclass
        y_prob = ensemble.predict_proba(X)
    
    # Compute predictions and metrics
    eval_results = _compute_predictions_and_metrics(
        all_probs=y_prob,
        all_true=y,
        sample_ids=sample_ids,
        task=task,
        target_labels=target_labels,
        output_dir=output_dir,
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
    """
    hyperparams = cfg["hyperparams"]
    task = cfg["task"]
    target_labels = cfg["target_labels"]
    
    if accelerator.is_main_process:
        logger.info(f"\n{'='*50}")
        logger.info("Multi-view Logistic Regression Ensemble")
        logger.info(f"View planes: {view_planes}")
        logger.info(f"{'='*50}")
    
    # Create consistent train/val split (same across all views)
    all_labels = np.array([train_datasets[view_planes[0]]._get_label(sid).numpy() for sid in train_datasets[view_planes[0]].sample_ids])
    val_split_ratio = hyperparams.get("val_split_ratio", 0.1)
    
    if task == "multilabel":
        n_splits = max(2, int(1.0 / val_split_ratio))
        kfold = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(kfold.split(X=np.arange(len(train_datasets[view_planes[0]])), y=all_labels))
        train_idx, val_idx = splits[0]
    else:
        train_idx, val_idx = train_test_split(
            np.arange(len(train_datasets[view_planes[0]])),
            test_size=val_split_ratio,
            stratify=all_labels,
            random_state=42,
        )
    
    if accelerator.is_main_process:
        logger.info(f"Training samples: {len(train_idx)}, Validation samples: {len(val_idx)}")
    
    # Step 1: Train single-view classifiers and collect predictions for each view
    view_predictions = {}
    
    for plane in view_planes:
        if accelerator.is_main_process:
            logger.info(f"\n{'='*50}")
            logger.info(f"Training single-view classifier for: {plane}")
            logger.info(f"{'='*50}")
        
        train_dataset_view_plane = train_datasets[plane]
        test_dataset_view_plane = test_datasets[plane]
        
        # Use the same train/val indices for consistency
        train_subset = Subset(train_dataset_view_plane, train_idx)
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
    
    # Setup optimizer (only train unfrozen parameters)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(hyperparams.get("lr", 1e-4)),
        weight_decay=1e-4
    )
    
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
    early_stopping = EarlyStopping(patience=patience, mode="max", min_delta=0.001) if patience else None
    
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
) -> Dict:
    """
    Evaluate multi-view classifier on test set.
    
    Returns predictions, metrics, and attention weights for interpretability.
    """
    task = cfg["task"]
    target_labels = cfg["target_labels"]
    
    # Prepare model and dataloader for distributed evaluation
    model, test_loader = accelerator.prepare(model, test_loader)
    model.eval()
    
    all_logits = []
    all_labels = []
    all_sample_ids = []
    all_attention = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating...", disable=not accelerator.is_main_process):
            outputs = model(batch["features"], batch["seq_lengths"])
            all_logits.append(outputs["logits"].detach())
            all_labels.append(batch["labels"].detach())
            all_sample_ids.extend(batch["sample_ids"])
            all_attention.append(outputs["attention_weights"].detach())
    
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
) -> Dict:
    """
    Run complete multi-view evaluation pipeline with attention-based aggregation.
    
    1. Create multi-view datasets
    2. Initialize MultiViewClassifier with attention aggregation
    3. Linear probe training
    4. Evaluate on test set
    """
    hyperparams = cfg["hyperparams"]
    linear_hyperparams = hyperparams.get("linear", hyperparams)
    attention_hyperparams = hyperparams.get("attention", {})
    task = cfg["task"]
    
    if accelerator.is_main_process:
        logger.info(f"\n{'='*50}")
        logger.info(f"Multi-view evaluation with attention aggregation")
        logger.info(f"View planes: {view_planes}")
        logger.info(f"{'='*50}")
    
    # Create train/val split
    all_labels = np.array([train_dataset._get_label(sid).numpy() for sid in train_dataset.sample_ids])
    val_split_ratio = hyperparams.get("val_split_ratio", 0.1)
    
    if task == "multilabel":
        n_splits = max(2, int(1.0 / val_split_ratio))
        kfold = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(kfold.split(X=np.arange(len(train_dataset)), y=all_labels))
        train_idx, val_idx = splits[0]
    else:
        train_idx, val_idx = train_test_split(
            np.arange(len(train_dataset)),
            test_size=val_split_ratio,
            stratify=all_labels,
            random_state=42,
        )
    
    if accelerator.is_main_process:
        logger.info(f"Training samples: {len(train_idx)}, Validation samples: {len(val_idx)}")
    
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(train_dataset, val_idx)
    
    # Create dataloaders
    train_loader = DataLoader(train_subset, batch_size=hyperparams["batch_size"], shuffle=True,
                              collate_fn=multiview_classifier_collate_fn, num_workers=hyperparams.get("num_workers", 0))
    val_loader = DataLoader(val_subset, batch_size=hyperparams["batch_size"], shuffle=False,
                            collate_fn=multiview_classifier_collate_fn, num_workers=hyperparams.get("num_workers", 0))
    test_loader = DataLoader(test_dataset, batch_size=hyperparams["batch_size"], shuffle=False,
                             collate_fn=multiview_classifier_collate_fn, num_workers=hyperparams.get("num_workers", 0))
    
    # Get dimensions
    num_classes = get_num_classes(task, cfg["target_labels"], train_dataset)
    input_dim = cobra_model.embed_dim
    
    if accelerator.is_main_process:
        logger.info(f"Num classes: {num_classes}, COBRA output dim: {input_dim}")
    
    # Initialize model
    model = MultiViewClassifier(
        cobra_model=cobra_model,
        view_planes=view_planes,
        input_dim=input_dim,
        num_classes=num_classes,
        classifier_hidden_dim=linear_hyperparams.get("hidden_dim", 512),
        classifier_dropout=linear_hyperparams.get("dropout", 0.5),
        attention_hidden_dim=attention_hyperparams.get("hidden_dim", 128),
        attention_dropout=attention_hyperparams.get("dropout", 0.1),
        freeze_cobra=True,
    )
    
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

    # Get view plane(s) and model names from config
    view_planes = cfg["feat_dataset"]["plane"]
    if args.fm_model_names:
        model_names = args.fm_model_names.split()  
    else:
        model_names = cfg["feat_dataset"]["model_name"]
    if isinstance(model_names, str):
        model_names = [model_names]
    use_multiview = len(view_planes) > 1

    # Create output directory
    target_labels = "_".join(cfg["target_labels"])
    if use_multiview:
        planes_str = "_".join(view_planes)
        output_dir = os.path.join(cfg["output_dir"], f"{target_labels}_multiview_{planes_str}_{CURR_TIME}")
    else:
        output_dir = os.path.join(cfg["output_dir"], f"{target_labels}_{view_planes[0]}_{CURR_TIME}")
    
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
    
    # Load pretrained COBRA model
    checkpoint_path = args.checkpoint_path if args.checkpoint_path else cfg["checkpoint_path"]
    
    if accelerator.is_main_process:
        logger.info("Loading pretrained COBRA model...")
        logger.info(f"Checkpoint path: {checkpoint_path}")
    
    pretrain_config_path = Path(checkpoint_path).parent / "config.yaml"
    with open(pretrain_config_path, "r") as f:
        pretrain_cfg = yaml.safe_load(f)
    cobra_cfg = pretrain_cfg["model"]["cobra"]
    
    cobra_model = load_pretrained_cobra(
        checkpoint_path=checkpoint_path,
        accelerator=accelerator,
        model_config=cobra_cfg,
        encoder_type=cfg["encoder_type"],
        fm_pooling=args.fm_pooling,
        sequence_encoder=args.sequence_encoder,  
        slice_pooling=args.slice_pooling, 
    )
    cobra_model = cobra_model.to(accelerator.device)
    cobra_model.eval()
    
    # Compute class weights
    class_weights = compute_class_weights_for_weighted_loss(
        cfg["train_annots"],
        cfg["target_labels"],
        cfg["task"],
        accelerator.device
    ) if args.weighted_loss else None
    
    # Run evaluation
    if use_multiview:
        # Multi-view with logistic regression ensemble
        if accelerator.is_main_process:
            logger.info(f"Using multi-view logistic regression ensemble")
            logger.info(f"View planes: {view_planes}")
        
        # Create datasets for all views once
        train_datasets = {}
        test_datasets = {}
        
        for plane in view_planes:
            train_datasets[plane] = FeatClassificationDataset(
                feat_dir=cfg["feat_dataset"]["feat_dir"],
                slice_encoder_models=model_names,
                view_plane=plane,
                split="train",
                annotations_path=cfg["train_annots"],
                task=cfg["task"],
                target_columns=cfg["target_labels"],
            )
            
            test_datasets[plane] = FeatClassificationDataset(
                feat_dir=cfg["feat_dataset"]["feat_dir"],
                slice_encoder_models=model_names,
                view_plane=plane,
                split="test",
                annotations_path=cfg["test_annots"],
                task=cfg["task"],
                target_columns=cfg["target_labels"],
            )
        
        eval_results = run_multiview_logistic_ensemble(
            cobra_model=cobra_model,
            train_datasets=train_datasets,
            test_datasets=test_datasets,
            cfg=cfg,
            accelerator=accelerator,
            output_dir=output_dir,
            view_planes=view_planes,
            class_weights=class_weights,
        )
    else:
        # Single-view
        view_plane = view_planes[0]
        
        # Create datasets
        train_dataset = FeatClassificationDataset(
            feat_dir=cfg["feat_dataset"]["feat_dir"],
            slice_encoder_models=model_names,
            view_plane=view_plane,
            split="train",
            annotations_path=cfg["train_annots"],
            task=cfg["task"],
            target_columns=cfg["target_labels"],
        )
        
        test_dataset = FeatClassificationDataset(
            feat_dir=cfg["feat_dataset"]["feat_dir"],
            slice_encoder_models=model_names,
            view_plane=view_plane,
            split="test",
            annotations_path=cfg["test_annots"],
            task=cfg["task"],
            target_columns=cfg["target_labels"],
        )
        
        if accelerator.is_main_process:
            logger.info(f"Using single-view inference with plane: {view_plane}")
            logger.info(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
        
        eval_results = run_single_view_evaluation(
            cobra_model=cobra_model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            cfg=cfg,
            accelerator=accelerator,
            output_dir=output_dir,
            class_weights=class_weights,
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
        "--fm-pooling",
        type=str,
        choices=["mean", "concat"],
        default="mean",
        help="Foundation model pooling method: 'mean' (average) or 'concat' (concatenate)."
        # NOTE: 'attention' pooling is not supported for linear probing since fm_attn weights 
        # are randomly initialized and frozen. Use 'attention' only when fine-tuning COBRA.
    )
    parser.add_argument(
        "--slice-pooling",
        type=str,
        choices=["abmil", "cls"],
        default=None,
    )
    args = parser.parse_args()
    main(args)
