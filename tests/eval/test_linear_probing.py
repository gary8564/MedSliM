"""
Smoke tests for linear probing helpers in ``linear_classifier.py``.
"""
import torch

from med_slim.eval.linear_classifier import (
    ClassifierHead,
    _compute_loss,
    _compute_metrics,
)
from med_slim.utils.metrics.linear import get_loss_criterion


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_classifier_head_binary():
    clf = ClassifierHead(input_dim=64, num_classes=2, hidden_dim=32).to(DEVICE)
    x = torch.randn(8, 64, device=DEVICE)
    out = clf(x)
    assert out.shape == (8, 1)


def test_classifier_head_multiclass():
    clf = ClassifierHead(input_dim=64, num_classes=5, hidden_dim=32).to(DEVICE)
    x = torch.randn(8, 64, device=DEVICE)
    out = clf(x)
    assert out.shape == (8, 5)


def test_compute_loss_binary():
    logits = torch.randn(8, 1, device=DEVICE)
    labels = torch.randint(0, 2, (8,), device=DEVICE).float()
    criterion = get_loss_criterion("binary")
    loss = _compute_loss(logits, labels, "binary", criterion)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_compute_metrics_binary():
    logits = torch.randn(16, 1, device=DEVICE)
    labels = torch.randint(0, 2, (16,), device=DEVICE).long()
    metrics = _compute_metrics(logits, labels, "binary", 2, DEVICE)
    assert "auroc" in metrics
