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

    # AUPRC
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

    # ROC-AUC
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

    # Confusion matrix (Youden's J threshold)
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

    logger.info(f"Label: {class_label}")
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
    Generate plots for multilabel classification.

    Produces:
        - Per-label ROC, PR, and 2-by-2 confusion matrix plots
        - Combined macro-averaged ROC curve with all per-label curves

    Note: An N-by-N confusion matrix is NOT produced for multilabel because
    labels are independent -- a sample can have multiple labels simultaneously.

    Args:
        y_true: Ground truth labels [N, num_labels]
        y_pred_prob: Predicted probabilities [N, num_labels]
        class_labels: List of label names
        output_dir: Directory to save plots

    Returns:
        Dictionary mapping label names to their metrics, plus "overall" macro averages
    """
    fontdict = {'fontsize': 12, 'fontweight': 'bold'}
    metrics = {}
    n_labels = len(class_labels)

    os.makedirs(output_dir, exist_ok=True)

    per_label_fpr = {}
    per_label_tpr = {}
    per_label_aurocs = []
    per_label_auprcs = []

    for i, cls in enumerate(class_labels):
        auprc, roc_auc = visualize_binary_metrics(
            y_true[:, i],
            y_pred_prob[:, i],
            output_dir,
            label=cls,
        )

        # Store ROC curve data for combined plot
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_prob[:, i])
        per_label_fpr[i] = fpr
        per_label_tpr[i] = tpr

        metrics[cls] = {
            "AUROC": float(roc_auc),
            "AUPRC": float(auprc),
        }
        per_label_aurocs.append(roc_auc)
        per_label_auprcs.append(auprc)

    # Combined macro-averaged ROC curve
    fpr_grid = np.linspace(0.0, 1.0, 1000)
    mean_tpr = np.zeros_like(fpr_grid)

    for i in range(n_labels):
        mean_tpr += np.interp(fpr_grid, per_label_fpr[i], per_label_tpr[i])

    mean_tpr /= n_labels
    macro_auroc = float(auc(fpr_grid, mean_tpr))
    macro_auprc = float(np.mean(per_label_auprcs))

    # Plot combined figure
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(
        fpr_grid,
        mean_tpr,
        label=f"Macro-average (AUC = {macro_auroc:.3f})",
        color="navy",
        linestyle=":",
        linewidth=3,
    )

    cmap = plt.get_cmap("tab10") if n_labels <= 10 else plt.get_cmap("tab20")
    for i, cls in enumerate(class_labels):
        roc_auc_i = metrics[cls]["AUROC"]
        ax.plot(
            per_label_fpr[i],
            per_label_tpr[i],
            label=f"{cls} (AUC = {roc_auc_i:.3f})",
            color=cmap(i),
            linewidth=2,
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel("False Positive Rate", fontdict=fontdict)
    ax.set_ylabel("True Positive Rate", fontdict=fontdict)
    ax.set_title(
        "Per-label ROC Curves (Macro-averaged)",
        fontdict={'fontsize': 14, 'fontweight': 'bold'},
    )
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "roc_multilabel_macro.png"), dpi=300)
    plt.close(fig)

    # Overall macro-averaged metrics
    metrics["overall"] = {
        "AUROC": macro_auroc,
        "AUPRC": macro_auprc,
    }
    logger.info(f"Overall macro-averaged AUROC: {macro_auroc:.4f}, AUPRC: {macro_auprc:.4f}")

    return metrics


def visualize_multiclass_metrics(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    class_labels: List[str],
    output_dir: str,
) -> Dict[str, Dict[str, float]]:
    """
    Generate plots for multiclass classification:
        - Overall N-by-N confusion matrix
        - Per-class one-vs-rest ROC, PR, and 2-by-2 confusion matrix plots
        - Combined macro-averaged OvR ROC curve with all per-class curves

    Args:
        y_true: Ground truth labels [N] (class indices)
        y_pred_prob: Predicted probabilities [N, num_classes]
        class_labels: List of class names
        output_dir: Directory to save plots

    Returns:
        Dictionary with per-class and overall metrics
    """
    fontdict = {'fontsize': 12, 'fontweight': 'bold'}
    metrics = {}
    n_classes = len(class_labels)

    os.makedirs(output_dir, exist_ok=True)

    # Predictions
    pred_idx = np.argmax(y_pred_prob, axis=1)
    
    # Overall confusion matrix
    cm = confusion_matrix(y_true, pred_idx)

    # Plot overall confusion matrix
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
    ax.set_xlabel("Predictions", fontdict=fontdict)
    ax.set_ylabel("Ground Truths", fontdict=fontdict)
    ax.set_title("Confusion Matrix", fontdict={'fontsize': 14, 'fontweight': 'bold'})
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300)
    plt.close(fig)

    # Per-class ROC curves (one-vs-rest)
    per_class_fpr = {}
    per_class_tpr = {}
    per_class_aurocs = []
    per_class_auprcs = []

    for i, cls in enumerate(class_labels):
        y_true_binary = (y_true == i).astype(int)
        y_pred_binary_prob = y_pred_prob[:, i]

        # Per-class plots (ROC, PR, 2×2 confusion matrix)
        auprc, roc_auc = visualize_binary_metrics(
            y_true_binary,
            y_pred_binary_prob,
            output_dir,
            label=cls,
        )

        # Store ROC curve data for combined plot
        fpr, tpr, _ = roc_curve(y_true_binary, y_pred_binary_prob)
        per_class_fpr[i] = fpr
        per_class_tpr[i] = tpr

        metrics[cls] = {
            "AUROC": float(roc_auc),
            "AUPRC": float(auprc)
        }
        per_class_aurocs.append(roc_auc)
        per_class_auprcs.append(auprc)

    # Combined macro-averaged ROC curve
    # Interpolate all per-class ROC curves with shared FPR grid points
    fpr_grid = np.linspace(0.0, 1.0, 1000)
    mean_tpr = np.zeros_like(fpr_grid)

    for i in range(n_classes):
        mean_tpr += np.interp(fpr_grid, per_class_fpr[i], per_class_tpr[i])

    mean_tpr /= n_classes
    macro_auroc = float(auc(fpr_grid, mean_tpr))
    macro_auprc = float(np.mean(per_class_auprcs))

    # Plot combined figure
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.plot(
        fpr_grid,
        mean_tpr,
        label=f"Macro-average (AUC = {macro_auroc:.3f})",
        color="navy",
        linestyle=":",
        linewidth=3,
    )

    cmap = plt.get_cmap("tab10") if n_classes <= 10 else plt.get_cmap("tab20")
    for i, cls in enumerate(class_labels):
        roc_auc_i = metrics[cls]["AUROC"]
        ax.plot(
            per_class_fpr[i],
            per_class_tpr[i],
            label=f"{cls} (AUC = {roc_auc_i:.3f})",
            color=cmap(i),
            linewidth=2,
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel("False Positive Rate", fontdict=fontdict)
    ax.set_ylabel("True Positive Rate", fontdict=fontdict)
    ax.set_title(
        "One-vs-Rest ROC Curves (Macro-averaged)",
        fontdict={'fontsize': 14, 'fontweight': 'bold'},
    )
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "roc_multiclass_macro.png"), dpi=300)
    plt.close(fig)

    # Overall macro-averaged metrics
    metrics["overall"] = {
        "AUROC": macro_auroc,
        "AUPRC": macro_auprc,
    }

    logger.info(f"Multiclass Confusion Matrix:\n{cm}")
    logger.info(f"Overall macro-averaged AUROC: {macro_auroc:.4f}, AUPRC: {macro_auprc:.4f}")

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
