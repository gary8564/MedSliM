"""
Smoke tests for linear probing helpers in ``linear_classifier.py``.
"""
import torch
import torch.nn as nn

from med_slim.eval.linear_classifier import (
    ClassifierHead,
    SingleViewClassifier,
    MultiViewClassifier,
    _compute_loss,
    _compute_metrics,
)
from med_slim.eval.load_cobra import resolve_eval_fm_ids
from med_slim.utils.metrics.linear import get_loss_criterion


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DummyCobra(nn.Module):
    def __init__(self, output_dim: int = 64):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))
        self.output_dim = output_dim
        self.last_physical_positions = None
        self.last_fm_ids = None
        self.calls = []

    def forward(self, features, seq_lengths=None, physical_positions=None, fm_ids=None, **_):
        self.last_physical_positions = physical_positions
        self.last_fm_ids = fm_ids
        self.calls.append(physical_positions)
        batch_size = features[0].shape[0]
        return torch.zeros(batch_size, self.output_dim, device=features[0].device)


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


def test_single_view_classifier_passes_physical_positions():
    cobra = DummyCobra(output_dim=64).to(DEVICE)
    model = SingleViewClassifier(
        cobra_model=cobra,
        input_dim=64,
        num_classes=2,
        freeze_cobra=True,
    ).to(DEVICE)
    features = [torch.randn(3, 5, 16, device=DEVICE)]
    seq_lengths = torch.tensor([5, 4, 3], device=DEVICE)
    physical_positions = torch.arange(5, device=DEVICE).float().expand(3, -1)

    out = model(features, seq_lengths, physical_positions)

    assert out["logits"].shape == (3, 1)
    assert cobra.last_physical_positions is not None
    assert torch.allclose(cobra.last_physical_positions, physical_positions)


def test_single_view_classifier_passes_fm_ids():
    cobra = DummyCobra(output_dim=64).to(DEVICE)
    model = SingleViewClassifier(
        cobra_model=cobra,
        input_dim=64,
        num_classes=2,
        freeze_cobra=True,
        fm_ids=[2, 0],
    ).to(DEVICE)
    features = [
        torch.randn(3, 5, 16, device=DEVICE),
        torch.randn(3, 5, 16, device=DEVICE),
    ]
    seq_lengths = torch.tensor([5, 4, 3], device=DEVICE)

    model(features, seq_lengths)

    assert torch.equal(cobra.last_fm_ids, torch.tensor([2, 0], device=DEVICE))


def test_multi_view_classifier_passes_physical_positions():
    cobra = DummyCobra(output_dim=64).to(DEVICE)
    view_planes = ["sagittal", "coronal"]
    model = MultiViewClassifier(
        cobra_model=cobra,
        view_planes=view_planes,
        input_dim=64,
        num_classes=2,
        freeze_cobra=True,
    ).to(DEVICE)
    features = {
        plane: [torch.randn(2, 4, 16, device=DEVICE)]
        for plane in view_planes
    }
    seq_lengths = {
        plane: torch.tensor([4, 3], device=DEVICE)
        for plane in view_planes
    }
    physical_positions = {
        plane: torch.arange(4, device=DEVICE).float().expand(2, -1)
        for plane in view_planes
    }

    out = model(features, seq_lengths, physical_positions)

    assert out["logits"].shape == (2, 1)
    assert len(cobra.calls) == len(view_planes)
    for plane, call in zip(view_planes, cobra.calls):
        assert torch.allclose(call, physical_positions[plane])


def test_resolve_eval_fm_ids_uses_checkpoint_order():
    pretrain_state = {"fm_id_order": ["dinov2", "rad-dino", "mri-core"]}
    fm_ids = resolve_eval_fm_ids(
        "router",
        ["mri-core", "dinov2"],
        pretrain_cfg={},
        pretrain_state=pretrain_state,
    )
    assert fm_ids == [2, 0]


def test_resolve_eval_fm_ids_ignores_non_router_pooling():
    fm_ids = resolve_eval_fm_ids(
        "avg_pool",
        ["mri-core"],
        pretrain_cfg={},
        pretrain_state=None,
    )
    assert fm_ids is None


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
