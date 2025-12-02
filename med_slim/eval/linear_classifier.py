"""
Linear Probing Evaluation for pretrained MedSliM SSL Model.
"""
import os
import argparse
import warnings
import yaml
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold, train_test_split
from tqdm import tqdm
from datetime import datetime
from accelerate import Accelerator
from typing import Dict, List, Optional, Tuple
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from transformers import get_cosine_schedule_with_warmup

from med_slim.model.sequence_encoder.cobra import Cobra
from med_slim.data.feat_dataset import FeatClassificationDataset, linear_classifier_collate_fn
from med_slim.eval.load_cobra import load_pretrained_cobra
from med_slim.eval.extract_feats import get_cobra_feats
from med_slim.utils.callbacks.early_stopping import EarlyStopping
from med_slim.utils.metrics.linear import get_loss_criterion, get_eval_metrics, get_num_classes
from med_slim.utils.viz.linear import compute_and_visualize_metrics
from med_slim.logging.setup import init_logging

init_logging()
logger = logging.getLogger(__name__)

CURR_TIME = datetime.now().strftime("%Y-%m-%d-%H:%M")


class LinearClassifier(nn.Module):
    """
    Linear classifier head for linear probing.
    """
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


def _compute_loss(logits: torch.Tensor, labels: torch.Tensor, task: str, criterion: nn.Module) -> torch.Tensor:
    """
    Compute task-specific loss.
    - BCEWithLogitsLoss (binary/multilabel): expects matching shapes and float targets
    - CrossEntropyLoss (multiclass): expects [B, C] logits, [B] long targets
    """
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
    """
    Compute metrics for training/validation monitoring using torchmetrics.
    Used during cross-validation phase for fast, incremental computation.

    For final test evaluation, use evaluate_classifier() which calls visualization.
    """
    metrics = get_eval_metrics(task, num_classes, device)
    logits = logits.to(device)
    labels = labels.to(device)

    if task == "binary":
        logits = logits.squeeze(-1)
        labels = labels.long()
    elif task == "multilabel":
        labels = labels.long()

    for metric in metrics.values():
        metric.update(logits, labels)

    return {name: metric.compute().item() for name, metric in metrics.items()}


def train_per_epoch(
    classifier: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    accelerator: Accelerator,
    task: str,
    num_classes: int,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> Tuple[float, Dict[str, float]]:
    classifier.train()

    total_loss = torch.tensor(0.0, device=accelerator.device)
    total_samples = torch.tensor(0.0, device=accelerator.device)
    logits_buffer: List[torch.Tensor] = []
    labels_buffer: List[torch.Tensor] = []

    for embeddings, labels in dataloader:
        optimizer.zero_grad()
        with accelerator.autocast():
            logits = classifier(embeddings)
            loss = _compute_loss(logits, labels, task, criterion)
        accelerator.backward(loss)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        batch_size = labels.shape[0]
        total_loss += loss.detach() * batch_size
        total_samples += batch_size
        logits_buffer.append(logits.detach())
        labels_buffer.append(labels.detach())

    gathered_loss = accelerator.reduce(total_loss, reduction="sum")
    gathered_samples = accelerator.reduce(total_samples, reduction="sum")
    avg_loss = (gathered_loss / gathered_samples).item()

    all_logits = accelerator.gather_for_metrics(torch.cat(logits_buffer))
    all_labels = accelerator.gather_for_metrics(torch.cat(labels_buffer))
    metrics = _compute_metrics(all_logits, all_labels, task, num_classes, accelerator.device)

    return avg_loss, metrics


@torch.no_grad()
def eval_per_epoch(
    classifier: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    accelerator: Accelerator,
    task: str,
    num_classes: int,
) -> Tuple[float, Dict[str, float]]:
    classifier.eval()

    total_loss = torch.tensor(0.0, device=accelerator.device)
    total_samples = torch.tensor(0.0, device=accelerator.device)
    logits_buffer: List[torch.Tensor] = []
    labels_buffer: List[torch.Tensor] = []

    for embeddings, labels in dataloader:
        logits = classifier(embeddings)
        loss = _compute_loss(logits, labels, task, criterion)

        batch_size = labels.shape[0]
        total_loss += loss.detach() * batch_size
        total_samples += batch_size
        logits_buffer.append(logits.detach())
        labels_buffer.append(labels.detach())

    gathered_loss = accelerator.reduce(total_loss, reduction="sum")
    gathered_samples = accelerator.reduce(total_samples, reduction="sum")
    avg_loss = (gathered_loss / gathered_samples).item()

    all_logits = accelerator.gather_for_metrics(torch.cat(logits_buffer))
    all_labels = accelerator.gather_for_metrics(torch.cat(labels_buffer))
    metrics = _compute_metrics(all_logits, all_labels, task, num_classes, accelerator.device)

    return avg_loss, metrics


def prepare_saving_metrics(
    computed_metrics: Dict[str, float],
    task: str,
    viz_metrics: Optional[Dict] = None
) -> Dict:
    result_metrics = {}

    if task == "binary":
        result_metrics = {
            "AUROC": float(computed_metrics.get("auroc", 0.0)),
            "AUPRC": float(computed_metrics.get("auprc", 0.0))
        }
    elif task == "multilabel":
        for class_name, class_metrics in viz_metrics.items():
            result_metrics[class_name] = {
                "AUROC": float(class_metrics.get("AUROC", 0.0)),
                "AUPRC": float(class_metrics.get("AUPRC", 0.0)),
            }

        result_metrics["overall"] = {
            "AUROC": float(computed_metrics.get("auroc", 0.0)),
            "AUPRC": float(computed_metrics.get("auprc", 0.0)),
        }
    else: 
        for class_name, class_vals in viz_metrics.items():
            result_metrics[class_name] = {
                "AUROC": float(class_vals.get("AUROC", 0.0)),
                "AUPRC": float(class_vals.get("AUPRC", 0.0)),
            }

        result_metrics["overall"] = {
            "AUROC": float(computed_metrics.get("auroc", 0.0)),
            "AUPRC": float(computed_metrics.get("auprc", 0.0)),
        }

    return result_metrics


def train_linear_classifier(
    classifier: LinearClassifier,
    train_embeddings: torch.Tensor,
    train_labels: torch.Tensor,
    val_embeddings: torch.Tensor,
    val_labels: torch.Tensor,
    cfg: Dict,
    accelerator: Accelerator,
    output_dir: str,
    fold: Optional[int] = None,
) -> Dict:
    """
    Train the linear classifier on extracted embeddings.
    
    Args:
        classifier: Linear classifier head
        train_embeddings: Training embeddings
        train_labels: Training labels
        val_embeddings: Validation embeddings
        val_labels: Validation labels
        cfg: Configuration
        accelerator: HuggingFace Accelerator
        output_dir: Directory to save checkpoints
        fold: Current fold number (for logging)
    
    Returns:
        classifier: Trained classifier
        metrics: evaluation metrics
    """
    hyperparams = cfg["hyperparams"]
    task = cfg["task"]
    num_classes = classifier.num_classes
    if task == "binary" and num_classes != 2:
        raise ValueError(f"Binary task requires 2 classes, got {num_classes}")
    elif task == "multiclass" and num_classes <= 2:
        raise ValueError(f"Multiclass task requires more than 2 classes, got {num_classes}")
    elif task == "multilabel" and num_classes <= 1:
        raise ValueError(f"Multilabel task requires more than 1 class, got {num_classes}")
    
    # Create tensor datasets
    train_dataset = torch.utils.data.TensorDataset(train_embeddings, train_labels)
    val_dataset = torch.utils.data.TensorDataset(val_embeddings, val_labels)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=hyperparams["batch_size"],
        shuffle=True,
        drop_last=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=hyperparams["batch_size"],
        shuffle=False,
        drop_last=False
    )
    
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=float(hyperparams["lr"]),
        weight_decay=1e-4
    )
    criterion = get_loss_criterion(task)
    
    # Create scheduler
    num_training_steps = len(train_loader) * hyperparams["max_epochs"]
    warmup_ratio = hyperparams.get("warmup_ratio", 0.1)
    num_warmup_steps = int(warmup_ratio * num_training_steps)
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Prepare with accelerator
    classifier, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        classifier, optimizer, train_loader, val_loader, scheduler
    )
    
    # Setup checkpoint path
    ckpt_dir = os.path.join(output_dir, "checkpoints")
    if accelerator.is_main_process:
        os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_name = f"classifier_fold{fold}.pt" if fold is not None else "final_classifier.pt"
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    
    # Initialize early stopping
    early_stopping = EarlyStopping(
        patience=hyperparams["patience"],
        mode="max",  # Maximize AURPC
        ckpt_path=ckpt_path,
        accelerator=accelerator
    )
    
    # Logging prefix for wandb
    log_prefix = f"fold_{fold}/" if fold is not None else ""
    
    for epoch in tqdm(
        range(hyperparams["max_epochs"]),
        desc="Training linear classifier...",
        disable=not accelerator.is_main_process,
    ):
        train_loss, train_metrics = train_per_epoch(
            classifier,
            train_loader,
            optimizer,
            criterion,
            accelerator,
            task,
            num_classes,
            scheduler,
        )

        val_loss, val_metrics = eval_per_epoch(
            classifier,
            val_loader,
            criterion,
            accelerator,
            task,
            num_classes,
        )
        val_auroc = val_metrics.get("auroc", 0.0)
        
        # Log to wandb
        if accelerator.is_main_process:
            current_lr = scheduler.get_last_lr()[0] if scheduler else cfg["hyperparams"]["lr"]
            log_dict = {
                f"{log_prefix}lr": current_lr,
                f"{log_prefix}train/train_loss": train_loss,
                f"{log_prefix}val/val_loss": val_loss,
                f"{log_prefix}epoch": epoch + 1,
            }
            for name, value in train_metrics.items():
                log_dict[f"{log_prefix}train/{name}"] = value
            for name, value in val_metrics.items():
                log_dict[f"{log_prefix}val/{name}"] = value
            wandb.log(log_dict)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_auroc={val_auroc:.4f}")
        
        # Early stopping check
        should_stop, best_score = early_stopping.step(
            val_score=val_auroc,
            model=classifier,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch
        )
        
        if should_stop:
            if accelerator.is_main_process:
                logger.info(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    classifier = early_stopping.load_best_model(accelerator.unwrap_model(classifier))
    
    return {"best_model": classifier, "best_val_auroc": early_stopping.best_score}


def evaluate_classifier(
    classifier: LinearClassifier,
    test_embeddings: torch.Tensor,
    test_labels: torch.Tensor,
    test_sample_ids: List,
    cfg: Dict,
    accelerator: Accelerator,
    output_dir: Optional[str] = None,
) -> Dict:
    """
    Evaluate the classifier on test set (final evaluation with visualization).

    For final evaluation, we only use visualization module which computes
    comprehensive metrics (AUROC, AUPRC) + generates plots. This avoids
    duplicate computation since visualization needs to compute curves anyway.

    Returns detailed metrics and predictions.
    """
    task = cfg["task"]
    num_classes = classifier.num_classes
    batch_size = cfg["hyperparams"]["batch_size"]

    test_dataset = torch.utils.data.TensorDataset(test_embeddings, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    test_loader, classifier = accelerator.prepare(test_loader, classifier)
    classifier.eval()

    # Collect predictions (no metric computation here - done by visualization)
    all_logits = []
    all_true = []

    with torch.no_grad():
        for x, y in test_loader:
            logits = classifier(x)
            all_logits.append(logits.cpu())
            all_true.append(y.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_true = torch.cat(all_true, dim=0).numpy()

    # Convert logits to probs/preds
    if task == "binary":
        all_probs = torch.sigmoid(all_logits.squeeze(-1)).numpy()
        all_preds = (all_probs > 0.5).astype(int)
    elif task == "multilabel":
        all_probs = torch.sigmoid(all_logits).numpy()
        all_preds = (all_probs > 0.5).astype(int)
    else:  # multiclass
        all_probs = F.softmax(all_logits, dim=-1).numpy()
        all_preds = all_logits.argmax(dim=-1).numpy()

    # Compute comprehensive metrics via visualization module
    # This computes AUROC, AUPRC and generates all plots
    viz_metrics = None
    if output_dir is not None:
        fig_dir = os.path.join(output_dir, "fig")
        class_labels = cfg["target_labels"]

        # For binary classification, use first target column as positive class label
        if task == "binary":
            viz_labels = class_labels[:1] if class_labels else ["positive"]
        else:
            viz_labels = class_labels

        viz_metrics = compute_and_visualize_metrics(
            y_true=all_true,
            y_pred_prob=all_probs,
            task=task,
            class_labels=viz_labels,
            output_dir=fig_dir,
            accelerator=accelerator
        )

        if accelerator.is_main_process:
            logger.info(f"Visualization plots saved to: {fig_dir}")

    if accelerator.is_main_process:
        final_metrics = {}
        if task == "binary":
            final_metrics["auroc"] = viz_metrics.get("AUROC", 0.0)
            final_metrics["auprc"] = viz_metrics.get("AUPRC", 0.0)
        elif task in ["multilabel", "multiclass"]:
            # For multilabel/multiclass, compute macro average
            auroc_values = [m.get("AUROC", 0.0) for m in viz_metrics.values()]
            auprc_values = [m.get("AUPRC", 0.0) for m in viz_metrics.values()]
            final_metrics["auroc"] = np.mean(auroc_values)
            final_metrics["auprc"] = np.mean(auprc_values)
            
    # Create results DataFrame
    if task == "binary":
        results_df = pd.DataFrame({
            "exam_id": test_sample_ids,
            "true_labels": all_true.tolist(),
            "pred_labels": all_preds.tolist(),
            "pred_probs": all_probs.tolist()
        })
    elif task == "multilabel":
        results_df = pd.DataFrame({"exam_id": test_sample_ids})
        results_df["true_labels"] = [list(map(int, row)) for row in all_true]
        results_df["pred_labels"] = [[cfg["target_labels"][i]
                                      for i, p in enumerate(row) if p >= 0.5]
                                     for row in all_probs]
        results_df["pred_probs"] = [list(row) for row in all_probs]
    else:  # multiclass
        results_df = pd.DataFrame({
            "exam_id": test_sample_ids,
            "true_labels": all_true.tolist(),
            "pred_labels": all_preds.tolist(),
            "pred_probs": [list(row) for row in all_probs]
        })

    return {
        "metrics": final_metrics,
        "predictions": results_df,
        "viz_metrics": viz_metrics
    }


def run_cross_validation(
    cobra_model: Cobra,
    train_dataset: FeatClassificationDataset,
    cfg: Dict,
    accelerator: Accelerator,
    output_dir: str,
) -> None:
    """
    Run stratified k-fold cross-validation on training set for performance estimation.
    """
    hyperparams = cfg["hyperparams"]
    task = cfg["task"]
    k_folds = hyperparams["k_folds"]
    
    # Get all labels for stratification
    all_labels = np.array([train_dataset._get_label(sid).numpy() for sid in train_dataset.sample_ids])
    if task == "multilabel":
        # MultilabelStratifiedKFold for proper multilabel stratification
        kfold = MultilabelStratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        splits = kfold.split(X=train_dataset.sample_ids, y=all_labels)
    else:
        # StratifiedKFold for binary/multiclass
        kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        splits = kfold.split(train_dataset.sample_ids, all_labels)
    
    cv_fold_results = []
    for fold, (train_idx, val_idx) in enumerate(splits):
        if accelerator.is_main_process:
            logger.info(f"\n{'='*50}")
            logger.info(f"Fold {fold + 1}/{k_folds}")
            logger.info(f"{'='*50}")

        # Create fold subsets
        train_subset = Subset(train_dataset, train_idx)
        val_subset = Subset(train_dataset, val_idx)

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

        # Extract COBRA embeddings
        train_embeddings, train_labels, _ = get_cobra_feats(cobra_model, train_loader, accelerator)
        val_embeddings, val_labels, _ = get_cobra_feats(cobra_model, val_loader, accelerator)

        # Determine num_classes
        num_classes = get_num_classes(task, cfg["target_labels"], train_dataset)

        # Initialize classifier
        input_dim = train_embeddings.shape[1]
        classifier = LinearClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dim=hyperparams["hidden_dim"],
            dropout=hyperparams["dropout"]
        )

        # Train classifier 
        training_results = train_linear_classifier(
            classifier,
            train_embeddings,
            train_labels,
            val_embeddings,
            val_labels,
            cfg,
            accelerator,
            output_dir,
            fold=fold,
        )
        fold_best_auroc = training_results["best_val_auroc"]

        if accelerator.is_main_process:
            logger.info(f"Fold {fold + 1} Best Validation AUROC: {fold_best_auroc:.4f}")
            cv_fold_results.append({
                "fold": fold + 1,
                "best_val_auroc": fold_best_auroc
            })
            
    
    # Aggregate cross-validation results
    cv_results = pd.DataFrame(cv_fold_results)
    mean_auroc = cv_results["best_val_auroc"].mean()
    std_auroc = cv_results["best_val_auroc"].std()
    
    if accelerator.is_main_process:
        logger.info(f"\n{'='*50}")
        logger.info(f"Cross-Validation Results")
        logger.info(f"{'='*50}")
        logger.info(f"Mean AUROC: {mean_auroc:.4f} ± {std_auroc:.4f}")
        
        # Log summary table
        wandb.run.summary["cross_val_mean_auroc"] = mean_auroc
        wandb.run.summary["cross_val_std_auroc"] = std_auroc
        

def run_final_evaluation(
    cobra_model: Cobra,
    train_dataset: FeatClassificationDataset,
    test_dataset: FeatClassificationDataset,
    cfg: Dict,
    accelerator: Accelerator,
    output_dir: str,
) -> Dict:
    """
    Train the final model on the entire training set and evaluate on test set after cross-validation has been used to estimate performance and validate
    hyperparameters.
    
    Returns:
        Dictionary containing evaluation results (metrics, predictions, viz_metrics)
    """
    hyperparams = cfg["hyperparams"]
    task = cfg["task"]
    
    if accelerator.is_main_process:
        logger.info(f"\n{'='*50}")
        logger.info("Training final model on entire training set and evaluating on test set")
        logger.info(f"{'='*50}")
    
    # Get all labels for creating a validation split
    all_labels = np.array([train_dataset._get_label(sid).numpy() for sid in train_dataset.sample_ids])
    sample_ids = np.array(train_dataset.sample_ids)
    
    # Create a validation split (e.g., 10% of training data) for early stopping
    val_split_ratio = hyperparams.get("val_split_ratio", 0.1)
    
    if task == "multilabel":
        # For multilabel, use MultilabelStratifiedKFold to create a single stratified split
        # We'll use a single fold from a k-fold split where k = 1/val_split_ratio
        n_splits = max(2, int(1.0 / val_split_ratio))
        kfold = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(kfold.split(X=sample_ids, y=all_labels))
        train_idx, val_idx = splits[0]  # Take first split
    else:
        # For binary/multiclass, use train_test_split for precise stratified split
        train_idx, val_idx = train_test_split(
            np.arange(len(train_dataset)),
            test_size=val_split_ratio,
            stratify=all_labels,
            random_state=42,
            shuffle=True
        )
    
    if accelerator.is_main_process:
        logger.info(f"Training samples: {len(train_idx)}, Validation samples: {len(val_idx)}")
    
    # Create train/val subsets
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(train_dataset, val_idx)
    
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
    
    # Extract COBRA embeddings
    train_embeddings, train_labels, _ = get_cobra_feats(cobra_model, train_loader, accelerator)
    val_embeddings, val_labels, _ = get_cobra_feats(cobra_model, val_loader, accelerator)
    
    # Determine num_classes
    num_classes = get_num_classes(task, cfg["target_labels"], train_dataset)
    
    # Initialize classifier
    input_dim = train_embeddings.shape[1]
    classifier = LinearClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=hyperparams["hidden_dim"],
        dropout=hyperparams["dropout"]
    )
    
    # Train linear classifier on entire training set and evaluate on test set
    training_results = train_linear_classifier(
        classifier,
        train_embeddings,
        train_labels,
        val_embeddings,
        val_labels,
        cfg,
        accelerator,
        output_dir
    )
    
    best_classifier = training_results["best_model"]
    best_val_auroc = training_results["best_val_auroc"]
    
    if accelerator.is_main_process:
        logger.info(f"Final Model Training Complete. Best Validation AUROC: {best_val_auroc:.4f}")
    
    # Evaluate on test set
    if accelerator.is_main_process:
        logger.info(f"\n{'='*50}")
        logger.info("Evaluating Final Model on Test Set")
        logger.info(f"{'='*50}")
    
    # Create test data loaders
    test_loader = DataLoader(
        test_dataset,
        batch_size=hyperparams["batch_size"],
        shuffle=False,
        collate_fn=linear_classifier_collate_fn,
        num_workers=hyperparams.get("num_workers", 0)
    )
    
    # Extract COBRA embeddings for test set
    test_embeddings, test_labels, test_sample_ids = get_cobra_feats(cobra_model, test_loader, accelerator)
    
    # Evaluate on test set
    eval_results = evaluate_classifier(
        best_classifier,
        test_embeddings,
        test_labels,
        test_sample_ids,
        cfg,
        accelerator,
        output_dir=output_dir
    )

    computed_metrics = eval_results["metrics"]
    viz_metrics = eval_results.get("viz_metrics")

    if accelerator.is_main_process:
        logger.info(f"Test dataset evaluation results:")
        logger.info(f"Test AUROC: {computed_metrics['auroc']:.4f}")
        logger.info(f"Test AUPRC: {computed_metrics['auprc']:.4f}")

        # Create table directory for metrics and predictions (consistent with rad_dino)
        table_dir = os.path.join(output_dir, "table")
        os.makedirs(table_dir, exist_ok=True)

        # Prepare and save comprehensive metrics
        save_metrics = prepare_saving_metrics(
            computed_metrics=computed_metrics,
            task=cfg["task"],
            viz_metrics=viz_metrics
        )
        metrics_path = os.path.join(table_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(save_metrics, f, indent=4)
        logger.info(f"Saved comprehensive metrics to: {metrics_path}")

        # Save predictions to table directory
        predictions_path = os.path.join(table_dir, "predictions.csv")
        eval_results["predictions"].to_csv(predictions_path, index=False)
        logger.info(f"Saved predictions to: {predictions_path}")

        # Log to wandb
        wandb.run.summary["test_auroc"] = computed_metrics["auroc"]
        wandb.run.summary["test_auprc"] = computed_metrics["auprc"]

    return eval_results


def main(args):
    """Main function for linear probing evaluation."""
    if not args.cross_val and not args.deploy:
        raise ValueError("Either --cross-val or --deploy must be specified.")
    
    if args.cross_val and args.deploy:
        warnings.warn("Both --cross-val and --deploy are specified. Only --deploy will be run.")
    
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Load config
    with open(args.pretrain_config, "r") as f:
        pretrain_cfg = yaml.safe_load(f)
    
    with open(args.linear_classifier_config, "r") as f:
        linear_classifier_cfg = yaml.safe_load(f)
        
    # Create output directory
    target_labels = "_".join(linear_classifier_cfg["target_labels"])
    view_plane = linear_classifier_cfg["feat_dataset"]["plane"]
    output_dir = os.path.join(
        linear_classifier_cfg["output_dir"],
        f"{target_labels}_{view_plane}_{CURR_TIME}"
    )
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        # Save config
        with open(os.path.join(output_dir, "config.yaml"), "w") as f:
            yaml.dump(linear_classifier_cfg, f)
    
    # Initialize wandb
    if accelerator.is_main_process:
        wandb.init(
            project="medslim-linear-probing",
            name=f"{linear_classifier_cfg['feat_dataset']['dataset_name']}_{target_labels}",
            config={
                "task": linear_classifier_cfg["task"],
                "target_labels": linear_classifier_cfg["target_labels"],
                "hyperparams": linear_classifier_cfg["hyperparams"],
            },
            dir=output_dir,
        )
    
    # Load pretrained COBRA model
    if accelerator.is_main_process:
        logger.info("Loading pretrained COBRA model...")
    
    # Get model config if available
    model_cfg = pretrain_cfg["model"]["cobra"]
    
    cobra_model = load_pretrained_cobra(
        checkpoint_path=linear_classifier_cfg["checkpoint_path"],
        accelerator=accelerator,
        encoder_type=linear_classifier_cfg["encoder_type"],
        input_dims=model_cfg["input_dims"]
    )
    cobra_model = cobra_model.to(accelerator.device)
    cobra_model.eval()
    
    # Freeze COBRA weights
    for param in cobra_model.parameters():
        param.requires_grad = False
    
    # Create datasets
    view_plane = "sagittal"
    
    train_dataset = FeatClassificationDataset(
        feat_dir=linear_classifier_cfg["feat_dataset"]["feat_dir"],
        slice_encoder_models=linear_classifier_cfg["feat_dataset"]["model_name"],
        view_plane=view_plane,
        split="train",
        annotations_path=linear_classifier_cfg["train_annots"],
        task=linear_classifier_cfg["task"],
        target_columns=linear_classifier_cfg["target_labels"],
    )
    
    test_dataset = FeatClassificationDataset(
        feat_dir=linear_classifier_cfg["feat_dataset"]["feat_dir"],
        slice_encoder_models=linear_classifier_cfg["feat_dataset"]["model_name"],
        view_plane=view_plane,
        split="test",
        annotations_path=linear_classifier_cfg["test_annots"],
        task=linear_classifier_cfg["task"],
        target_columns=linear_classifier_cfg["target_labels"],
    )
    
    if accelerator.is_main_process:
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Test dataset size: {len(test_dataset)}")
    
    # Run cross-validation on training set for performance estimation
    if args.cross_val:
        run_cross_validation(
            cobra_model,
            train_dataset,
            linear_classifier_cfg,
            accelerator,
            output_dir
        )

    # Train final model on entire training set and evaluate on test set
    test_results = None
    if args.deploy:
        test_results = run_final_evaluation(
            cobra_model,
            train_dataset,
            test_dataset,
            linear_classifier_cfg,
            accelerator,
            output_dir
        )

    if accelerator.is_main_process:
        logger.info(f"\n{'='*50}")
        logger.info("Linear Probing Evaluation Complete!")
        logger.info(f"{'='*50}")
        if test_results is not None:
            logger.info(f"Test AUROC: {test_results['metrics']['auroc']:.4f}")
            logger.info(f"Test AUPRC: {test_results['metrics']['auprc']:.4f}")
        logger.info(f"Results saved to: {output_dir}")

        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear Probing Evaluation for MedSliM")
    parser.add_argument(
        "--pretrain-config",
        type=str,
        required=True,
        help="Path to the pretrained MedSliM config file"
    )
    parser.add_argument(
        "--linear-classifier-config",
        type=str,
        required=True,
        help="Path to the linear classifier config file"
    )
    parser.add_argument(
        "--cross-val",
        action="store_true",
        help="Run cross-validation for hyperparameter tuning"
    )
    parser.add_argument(
        "--deploy",
        action="store_true",
        help="Run the final evaluation and deploy the model"
    )
    args = parser.parse_args()
    main(args)
