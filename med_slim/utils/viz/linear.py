"""
Visualization utilities for classification results (ROC curves, confusion matrices, etc.)
"""
import os
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Tuple, Optional, Dict, List
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_recall_curve

from med_slim.logging.setup import init_logging

init_logging()
logger = logging.getLogger(__name__)


def visualize_binary_metrics(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    output_dir: str,
    label: Optional[str] = None
) -> Tuple[float, float]:
    """
    Generate ROC curve, PR curve, and confusion matrix plots for binary classification.

    Args:
        y_true: Ground truth binary labels
        y_pred_prob: Predicted probabilities
        output_dir: Directory to save plots
        label: Optional label for the plots (e.g., disease name)

    Returns:
        Tuple of (AUPRC, ROC_AUC)
    """
    fontdict = {'fontsize': 10, 'fontweight': 'bold'}

    if label is None:
        class_label = ""
        title = ""
        filename = ""
    else:
        class_label = label
        title = f"for {label}"
        filename = f"_{label.replace(' ', '_')}"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------- AUPRC ---------------------------------
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    auprc = auc(recall, precision)

    fig, axis_auprc = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    axis_auprc.plot(recall, precision, label=f"AP {class_label} = {auprc:.3f}")
    axis_auprc.set_xlim([0.0, 1.0])
    axis_auprc.set_ylim([0.0, 1.0])
    axis_auprc.set_xlabel("Recall", fontdict=fontdict)
    axis_auprc.set_ylabel("Precision", fontdict=fontdict)
    axis_auprc.set_title(f"PR Curve {title}", fontdict=fontdict)
    axis_auprc.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"auprc{filename}.png"), dpi=300)
    plt.close(fig)

    # ------------------------------- ROC-AUC ---------------------------------
    fprs, tprs, thresholds = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fprs, tprs)

    fig, axis_roc = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    axis_roc.plot(fprs, tprs, label=f"AUC {class_label} = {roc_auc:.3f}")
    axis_roc.plot([0, 1], [0, 1], 'k--')
    axis_roc.set_xlim([0.0, 1.0])
    axis_roc.set_ylim([0.0, 1.0])
    axis_roc.set_xlabel('False Positive Rate', fontdict=fontdict)
    axis_roc.set_ylabel('True Positive Rate', fontdict=fontdict)
    axis_roc.set_title(f'ROC Curve {title}', fontdict=fontdict)
    axis_roc.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"roc{filename}.png"), dpi=300)
    plt.close(fig)

    # -------------------------- Confusion Matrix -------------------------
    # Youden’s J to pick a threshold
    youden = tprs - fprs
    best_idx = youden.argmax()
    best_thr = thresholds[best_idx]
    logger.info(f"Best threshold: {best_thr:.3f}")
    y_pred = (y_pred_prob >= best_thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    # Handle edge cases where a class is not present
    sens = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if cm.shape[0] > 1 and (cm[1, 1] + cm[1, 0]) > 0 else 0
    spec = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if cm.shape[1] > 1 and (cm[0, 0] + cm[0, 1]) > 0 else 0

    df_cm = pd.DataFrame(cm, columns=['Negative', 'Positive'], index=['Negative', 'Positive'])
    fig, axis_cm = plt.subplots(1, 1, figsize=(5, 5))
    sns.heatmap(df_cm, ax=axis_cm, cbar=False, fmt='d', annot=True, cmap='Blues')
    axis_cm.set_title(f'Confusion Matrix {title}\nACC={acc:.3f}', fontdict=fontdict)
    axis_cm.set_xlabel('Predicted', fontdict=fontdict)
    axis_cm.set_ylabel('Ground Truth', fontdict=fontdict)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f"confusion_matrix{filename}.png"), dpi=300)
    plt.close(fig)

    logger.info(f"------Label {class_label}--------")
    logger.info(f"Number of positive samples: {np.sum(y_true)}")
    logger.info(f"Confusion Matrix:\n{cm}")
    logger.info(f"Sensitivity: {sens:.3f}")
    logger.info(f"Specificity: {spec:.3f}")
    logger.info(f"AUPRC: {auprc:.3f}")
    logger.info(f"ROC-AUC: {roc_auc:.3f}")

    return auprc, roc_auc


def visualize_multilabel_metrics(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    class_labels: List[str],
    output_dir: str,
) -> Dict[str, Dict[str, float]]:
    """
    Generate plots for multilabel classification (one set of plots per label).

    Args:
        y_true: Ground truth labels [N, num_classes]
        y_pred_prob: Predicted probabilities [N, num_classes]
        class_labels: List of class names
        output_dir: Directory to save plots

    Returns:
        Dictionary mapping class names to their metrics
    """
    metrics = {}

    for i, cls in enumerate(class_labels):
        auprc, roc_auc = visualize_binary_metrics(
            y_true[:, i],
            y_pred_prob[:, i],
            output_dir,
            label=cls
        )
        metrics[cls] = {
            "AUROC": float(roc_auc),
            "AUPRC": float(auprc)
        }

    return metrics


def visualize_multiclass_metrics(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    class_labels: List[str],
    output_dir: str,
) -> Dict[str, Dict[str, float]]:
    """
    Generate plots for multiclass classification.

    Args:
        y_true: Ground truth labels [N] (class indices)
        y_pred_prob: Predicted probabilities [N, num_classes]
        class_labels: List of class names
        output_dir: Directory to save plots

    Returns:
        Dictionary with per-class and overall metrics
    """
    metrics = {}

    os.makedirs(output_dir, exist_ok=True)

    # Predictions
    pred_idx = np.argmax(y_pred_prob, axis=1)

    # Overall confusion matrix
    cm = confusion_matrix(y_true, pred_idx)

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        ax=ax,
        xticklabels=class_labels,
        yticklabels=class_labels,
        cmap='Blues'
    )
    ax.set_xlabel("Predicted", fontdict={'fontsize': 12, 'fontweight': 'bold'})
    ax.set_ylabel("Ground Truth", fontdict={'fontsize': 12, 'fontweight': 'bold'})
    ax.set_title("Multiclass Confusion Matrix", fontdict={'fontsize': 14, 'fontweight': 'bold'})
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
    plt.close(fig)

    # Per-class ROC curves (one-vs-rest)
    for i, cls in enumerate(class_labels):
        y_true_binary = (y_true == i).astype(int)
        y_pred_binary_prob = y_pred_prob[:, i]

        auprc, roc_auc = visualize_binary_metrics(
            y_true_binary,
            y_pred_binary_prob,
            output_dir,
            label=cls
        )

        metrics[cls] = {
            "AUROC": float(roc_auc),
            "AUPRC": float(auprc)
        }

    logger.info(f"Multiclass Confusion Matrix:\n{cm}")

    return metrics


def compute_and_visualize_metrics(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    task: str,
    class_labels: List[str],
    output_dir: str,
) -> Dict:
    """
    Main entry point for computing and visualizing classification metrics.

    Args:
        y_true: Ground truth labels
        y_pred_prob: Predicted probabilities
        task: Task type ('binary', 'multiclass', 'multilabel')
        class_labels: List of class labels
        output_dir: Directory to save plots

    Returns:
        Dictionary containing computed metrics
    """
    logger.info(f"\nGenerating visualization plots for {task} classification...")
    logger.info(f"Saving plots to: {output_dir}")

    if task == "binary":
        label = class_labels[0] if class_labels else None
        auprc, roc_auc = visualize_binary_metrics(
            y_true,
            y_pred_prob,
            output_dir,
            label=label
        )
        return {
            "AUROC": float(roc_auc),
            "AUPRC": float(auprc)
        }
    elif task == "multilabel":
        return visualize_multilabel_metrics(
            y_true,
            y_pred_prob,
            class_labels,
            output_dir,
        )
    elif task == "multiclass":
        return visualize_multiclass_metrics(
            y_true,
            y_pred_prob,
            class_labels,
            output_dir,
        )
    else:
        raise ValueError(f"Unsupported task type: {task}. Must be one of ['binary', 'multiclass', 'multilabel']")
