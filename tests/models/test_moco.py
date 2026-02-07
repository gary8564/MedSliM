import torch
import pytest

from med_slim.model.ssl import MoCo


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Self-supervised contrastive learning tests
def test_moco_mamba2_forward_loss():
    """Test MoCo with mamba2 encoder and ABMIL pooling."""
    batch_size = 4
    num_slices = 12
    input_dim = 768
    embed_dim = 768
    contrast_dim = 128

    model = MoCo(
        embed_dim=embed_dim,
        contrast_dim=contrast_dim,
        input_dims=[768],
        num_heads=2,
        num_layers=1,
        T=0.2,
        dropout=0.0,
        d_state=32,
        att_dim=64,
    ).to(DEVICE).eval()

    x1 = torch.randn(batch_size, num_slices, input_dim, device=DEVICE)
    x2 = torch.randn(batch_size, num_slices, input_dim, device=DEVICE)
    with torch.no_grad():
        loss = model(x1, x2, input_feature_dims_1=None, input_feature_dims_2=None, m=0.99)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # scalar tensor
    assert torch.isfinite(loss).all()


def test_moco_transformer_forward_loss():
    """Test MoCo with transformer encoder and CLS pooling."""
    batch_size = 4
    num_slices = 12
    input_dim = 768

    model = MoCo(
        embed_dim=input_dim,
        contrast_dim=128,
        input_dims=[input_dim],
        num_heads=4,
        num_layers=1,
        T=0.2,
        dropout=0.0,
        sequence_encoder="transformer",
        pooling="cls",
    ).to(DEVICE).eval()

    x1 = torch.randn(batch_size, num_slices, input_dim, device=DEVICE)
    x2 = torch.randn(batch_size, num_slices, input_dim, device=DEVICE)
    seq_lens = torch.full((batch_size,), num_slices, dtype=torch.long, device=DEVICE)
    
    with torch.no_grad():
        loss = model(x1, x2, seq_lengths=seq_lens, m=0.99)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss).all()


# Semi-supervised contrastive learning tests
def _build_model(embed_dim=256, contrast_dim=64, input_dim=64):
    """Helper to create a small MoCo model for testing semi-supervised contrastive learning."""
    return MoCo(
        embed_dim=embed_dim,
        contrast_dim=contrast_dim,
        input_dims=[input_dim],
        num_heads=2,
        num_layers=1,
        T=0.2,
        dropout=0.0,
        d_state=32,
        att_dim=32,
    ).to(DEVICE).eval()


def test_moco_semi_supervised_forward():
    """Test MoCo forward with multi-label supervision (SemiSupCon mode)."""
    batch_size = 4
    num_slices = 8
    input_dim = 64

    model = _build_model(input_dim=input_dim)

    x1 = torch.randn(batch_size, num_slices, input_dim, device=DEVICE)
    x2 = torch.randn(batch_size, num_slices, input_dim, device=DEVICE)
    
    # Multi-label: [abnormal, acl, meniscus]
    labels = torch.tensor([
        [1, 0, 0],  # abnormal only
        [1, 1, 0],  # abnormal + ACL
        [0, 0, 0],  # healthy
        [1, 0, 1],  # abnormal + meniscus
    ], dtype=torch.float32, device=DEVICE)
    has_label = torch.tensor([True, True, True, False], device=DEVICE)

    with torch.no_grad():
        loss = model(
            x1, x2,
            input_feature_dims_1=None, input_feature_dims_2=None,
            m=0.99,
            labels=labels,
            has_label=has_label,
        )

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss).all()
    assert loss.item() > 0


def test_moco_semi_supervised_all_unlabeled_matches_infonce():
    """When all has_label=False, the loss should match pure InfoNCE."""
    torch.manual_seed(42)
    batch_size = 4
    num_slices = 8
    input_dim = 64
    num_classes = 3

    model = _build_model(input_dim=input_dim)

    x1 = torch.randn(batch_size, num_slices, input_dim, device=DEVICE)
    x2 = torch.randn(batch_size, num_slices, input_dim, device=DEVICE)

    # All unlabeled: should fall back to standard InfoNCE
    labels = torch.zeros(batch_size, num_classes, dtype=torch.float32, device=DEVICE)
    has_label = torch.zeros(batch_size, dtype=torch.bool, device=DEVICE)

    with torch.no_grad():
        loss_semisup = model(x1, x2, m=0.99, labels=labels, has_label=has_label)
        loss_infonce = model(x1, x2, m=0.99)  # No labels → pure InfoNCE

    assert torch.allclose(loss_semisup, loss_infonce, atol=1e-5), \
        f"All-unlabeled semi-supervised loss ({loss_semisup.item():.6f}) should match " \
        f"pure InfoNCE ({loss_infonce.item():.6f})"


def test_moco_semi_supervised_labels_affect_loss():
    """Verify that providing labels actually changes the loss value."""
    torch.manual_seed(42)
    batch_size = 8
    num_slices = 8
    input_dim = 64

    model = _build_model(input_dim=input_dim)

    x1 = torch.randn(batch_size, num_slices, input_dim, device=DEVICE)
    x2 = torch.randn(batch_size, num_slices, input_dim, device=DEVICE)

    # Create labels where multiple samples share the same class
    labels = torch.tensor([
        [1, 0, 0], [1, 0, 0],  # Two samples sharing 'abnormal'
        [0, 1, 0], [0, 1, 0],  # Two samples sharing 'ACL'
        [0, 0, 1], [0, 0, 1],  # Two samples sharing 'meniscus'
        [0, 0, 0], [0, 0, 0],  # Two healthy samples
    ], dtype=torch.float32, device=DEVICE)
    has_label = torch.ones(batch_size, dtype=torch.bool, device=DEVICE)

    with torch.no_grad():
        loss_infonce = model(x1, x2, m=0.99)
        loss_semisup = model(x1, x2, m=0.99, labels=labels, has_label=has_label)

    # With shared classes creating additional positives, the loss should differ
    assert not torch.allclose(loss_infonce, loss_semisup, atol=1e-5), \
        "Semi-supervised loss should differ from InfoNCE when labels create additional positives"


def test_moco_semi_supervised_mixed_labeled_unlabeled():
    """Test with a mix of labeled and unlabeled samples in the same batch."""
    torch.manual_seed(42)
    batch_size = 6
    num_slices = 8
    input_dim = 64

    model = _build_model(input_dim=input_dim)

    x1 = torch.randn(batch_size, num_slices, input_dim, device=DEVICE)
    x2 = torch.randn(batch_size, num_slices, input_dim, device=DEVICE)

    labels = torch.tensor([
        [1, 0, 0],  # labeled: abnormal
        [1, 0, 0],  # labeled: abnormal (same class → positive pair)
        [0, 1, 0],  # labeled: ACL
        [0, 0, 0],  # labeled: healthy
        [0, 0, 0],  # labeled: healthy (same → positive pair)
        [0, 0, 0],  # unlabeled (has_label=False, should use InfoNCE)
    ], dtype=torch.float32, device=DEVICE)
    has_label = torch.tensor([True, True, True, True, True, False], device=DEVICE)

    with torch.no_grad():
        loss = model(x1, x2, m=0.99, labels=labels, has_label=has_label)

    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0
    assert torch.isfinite(loss).all()
    assert loss.item() > 0


def test_moco_build_label_positive_mask():
    """Test the label positive mask construction directly."""
    model = _build_model()

    # 4 local samples, 4 global samples (single process)
    labels = torch.tensor([
        [1, 0, 0],  # sample 0: abnormal
        [1, 1, 0],  # sample 1: abnormal + ACL
        [0, 0, 0],  # sample 2: healthy
        [0, 0, 1],  # sample 3: meniscus
    ], dtype=torch.float32)
    has_label = torch.tensor([True, True, True, True])

    mask = model._build_label_positive_mask(labels, has_label, labels, has_label)

    # Sample 0 (abnormal) should be positive with sample 1 (abnormal+ACL) — shared 'abnormal'
    assert mask[0, 1] == 1.0
    assert mask[1, 0] == 1.0

    # Sample 0 should NOT be positive with sample 2 (healthy) or sample 3 (meniscus only)
    assert mask[0, 2] == 0.0
    assert mask[0, 3] == 0.0

    # Sample 2 (healthy) should be positive with itself (both-negative → 1.0)
    assert mask[2, 2] == 1.0

    # Sample 2 should NOT be positive with any non-healthy sample
    assert mask[2, 0] == 0.0
    assert mask[2, 1] == 0.0
    assert mask[2, 3] == 0.0

    # When one sample is unlabeled, no label-based positives
    has_label_mixed = torch.tensor([True, True, True, False])
    mask_mixed = model._build_label_positive_mask(labels, has_label_mixed, labels, has_label_mixed)

    # Sample 3 is unlabeled → no label-based positives with anyone
    assert mask_mixed[3, :].sum() == 0.0
    # And no one should be positive with sample 3
    assert mask_mixed[:, 3].sum() == 0.0

    # But labeled samples still have their positives
    assert mask_mixed[0, 1] == 1.0
    assert mask_mixed[2, 2] == 1.0


def test_moco_semi_supervised_backward():
    """Test that gradients flow correctly with semi-supervised loss."""
    torch.manual_seed(42)
    batch_size = 4
    num_slices = 8
    input_dim = 64

    model = _build_model(input_dim=input_dim)
    model.train()

    x1 = torch.randn(batch_size, num_slices, input_dim, device=DEVICE)
    x2 = torch.randn(batch_size, num_slices, input_dim, device=DEVICE)
    labels = torch.tensor([
        [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0],
    ], dtype=torch.float32, device=DEVICE)
    has_label = torch.ones(batch_size, dtype=torch.bool, device=DEVICE)

    loss = model(x1, x2, m=0.99, labels=labels, has_label=has_label)
    loss.backward()

    # Check that gradients exist for base encoder parameters
    for name, param in model.base_encoder.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            break  # Just check one parameter

