import torch
from torchmetrics import Accuracy, AUROC, AveragePrecision, F1Score
from typing import List, Optional

def get_num_classes(task: str, target_columns: List[str], dataset=None) -> int:
    """
    Determine number of classes based on task type.

    Args:
        task: Task type (binary, multiclass, multilabel)
        target_columns: List of target column names
        dataset: Optional dataset for multiclass label counting

    Returns:
        Number of classes
    """
    if task == "binary":
        return 2
    elif task == "multilabel":
        return len(target_columns)
    else:  # multiclass
        if dataset is None:
            raise ValueError("Dataset required for multiclass task to determine num_classes")
        return len(dataset.df_labels[target_columns[0]].unique())


def get_loss_criterion(task: str):
    """Get appropriate loss function based on task type."""
    criterion_map = {
        "multiclass": torch.nn.CrossEntropyLoss(),
        "multilabel": torch.nn.BCEWithLogitsLoss(),
        "binary": torch.nn.BCEWithLogitsLoss(),
        "regression": torch.nn.MSELoss()
    }
    if task not in criterion_map:
        raise NotImplementedError(f"Task {task} is not supported")
    return criterion_map[task]

def get_eval_metrics(task: str, num_classes: int, device: str):
    """Create appropriate metrics based on task type."""
    metrics = {}
    
    if task == "multiclass":
        metrics.update({
            "acc": Accuracy(task="multiclass", num_classes=num_classes),
            "auroc": AUROC(task="multiclass", num_classes=num_classes, average="macro"),
            "auprc": AveragePrecision(task="multiclass", num_classes=num_classes, average="macro"),
            "f1_score": F1Score(task="multiclass", num_classes=num_classes)
        })
    elif task == "multilabel":
        metrics.update({
            "acc": Accuracy(task="multilabel", num_labels=num_classes),
            "auroc": AUROC(task="multilabel", num_labels=num_classes, average="macro"),
            "auprc": AveragePrecision(task="multilabel", num_labels=num_classes, average="macro"),
            "f1_score": F1Score(task="multilabel", num_labels=num_classes)
        })
    elif task == "binary":
        metrics.update({
            "acc": Accuracy(task="binary"),
            "auroc": AUROC(task="binary"),
            "auprc": AveragePrecision(task="binary"),
            "f1_score": F1Score(task="binary")
        })
    
    return {k: v.to(device) for k, v in metrics.items() if v is not None}