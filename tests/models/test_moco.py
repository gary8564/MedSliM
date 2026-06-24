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


def test_moco_rejects_cross_attention_pooling():
    """Cross-attention pooling was removed from the MedSliM SSL pipeline."""
    with pytest.raises(AssertionError, match="Invalid slice_pooling"):
        MoCo(
            embed_dim=768,
            contrast_dim=128,
            input_dims=[768],
            num_heads=4,
            num_layers=1,
            T=0.2,
            dropout=0.0,
            pooling="cross_attention",
            d_state=32,
        )


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


# Subset FM mode (router) MoCo losses
def _build_subset_moco(fm_pooling="router", num_fms=4, **kwargs):
    return MoCo(
        embed_dim=64,
        contrast_dim=32,
        input_dims=[64, 128],
        num_heads=4,
        num_layers=1,
        T=0.2,
        dropout=0.0,
        sequence_encoder="transformer",
        pooling="abmil",
        fm_pooling=fm_pooling,
        num_fms=num_fms,
        att_dim=32,
        **kwargs,
    ).to(DEVICE)


def _subset_views(batch_size, k_sub, num_slices, max_feature_dim=128, num_regions=None):
    dims = [[64, 128][i % 2] for i in range(k_sub)]
    shape = (
        (batch_size, k_sub, num_slices, max_feature_dim)
        if num_regions is None
        else (batch_size, k_sub, num_slices, num_regions, max_feature_dim)
    )
    x1 = torch.randn(*shape, device=DEVICE)
    x2 = torch.randn(*shape, device=DEVICE)
    feat_dims = torch.tensor([dims] * batch_size, dtype=torch.long, device=DEVICE)
    # Controlled-overlap non-identical FM ID sets.
    fm_ids_1 = torch.tensor([[0, 1]] * batch_size, dtype=torch.long, device=DEVICE)
    fm_ids_2 = torch.tensor([[0, 2]] * batch_size, dtype=torch.long, device=DEVICE)
    return x1, x2, feat_dims, fm_ids_1, fm_ids_2


def test_moco_subset_router_returns_loss_dict():
    batch_size, k_sub, num_slices = 4, 2, 8
    model = _build_subset_moco(fm_pooling="router").eval()
    x1, x2, dims, ids1, ids2 = _subset_views(batch_size, k_sub, num_slices)

    with torch.no_grad():
        out = model(
            x1, x2,
            input_feature_dims_1=dims, input_feature_dims_2=dims,
            fm_ids_1=ids1, fm_ids_2=ids2, m=0.99,
        )
    assert isinstance(out, dict)
    assert "loss" in out and "loss_router_balance" in out and "loss_infonce" in out
    assert torch.isfinite(out["loss"]).all()
    assert out["loss"].ndim == 0


def test_moco_subset_router_flattened_regional_tokens():
    batch_size, k_sub, num_slices, regional_tokens = 4, 2, 6, 4
    num_regions = 1 + regional_tokens
    model = _build_subset_moco(
        fm_pooling="router",
        regional_tokens=regional_tokens,
    ).eval()
    x1, x2, dims, ids1, ids2 = _subset_views(
        batch_size, k_sub, num_slices, num_regions=num_regions
    )
    seq_lengths = torch.tensor([num_slices, 5, 4, 3], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        out = model(
            x1, x2,
            input_feature_dims_1=dims, input_feature_dims_2=dims,
            fm_ids_1=ids1, fm_ids_2=ids2,
            seq_lengths=seq_lengths,
            m=0.99,
        )

    assert isinstance(out, dict)
    assert torch.isfinite(out["loss"]).all()
    assert out["fm_usage"].shape[0] == model.num_fms


def test_moco_load_balance_targets_subset_size_not_batch_active_fms():
    """Balanced 2-FM subsets should have zero loss even when 3 distinct FMs appear."""
    model = _build_subset_moco(fm_pooling="router", num_fms=3).eval()
    weights = torch.full((2, 4, 2), 0.5, device=DEVICE)
    fm_ids = torch.tensor([[0, 1], [0, 2]], dtype=torch.long, device=DEVICE)

    loss = model._fm_load_balance_loss(weights, fm_ids)

    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)


def test_moco_aggregate_fm_usage_uniform():
    model = _build_subset_moco(fm_pooling="router", num_fms=2).eval()
    weights = torch.full((1, 3, 2), 0.5, device=DEVICE)
    fm_ids = torch.tensor([[0, 1]], dtype=torch.long, device=DEVICE)
    usage = MoCo.aggregate_fm_usage(weights, fm_ids, num_fms=2)
    assert usage.shape == (2,)
    assert torch.allclose(usage, torch.tensor([0.5, 0.5], device=DEVICE), atol=1e-6)


def test_moco_fm_usage():
    model = _build_subset_moco(fm_pooling="router", num_fms=2).eval()
    weights = torch.tensor([[[0.75, 0.25]]], dtype=torch.float32, device=DEVICE)
    fm_ids = torch.tensor([[0, 1]], dtype=torch.long, device=DEVICE)
    stats = {
        "fm_weights_local": weights,
        "fm_ids": fm_ids,
        "router_logits": torch.zeros(1, 1, 2, device=DEVICE),
        "fm_weight_entropy": torch.zeros(1, 1, device=DEVICE),
    }
    collected = model._collect_router_losses([stats, stats])
    assert "fm_usage" in collected
    assert collected["fm_usage"].shape == (2,)
    assert collected["fm_usage"][0].item() == pytest.approx(0.75, rel=1e-5)


def test_moco_load_balance_ignores_padded_slices():
    model = _build_subset_moco(fm_pooling="router", num_fms=2).eval()
    weights = torch.tensor(
        [
            [[0.5, 0.5], [0.5, 0.5], [1.0, 0.0], [1.0, 0.0]],
            [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]],
        ],
        dtype=torch.float32,
        device=DEVICE,
    )
    fm_ids = torch.tensor([[0, 1], [0, 1]], dtype=torch.long, device=DEVICE)
    slice_mask = torch.tensor(
        [[True, True, False, False], [True, True, True, True]],
        device=DEVICE,
    )

    loss = model._fm_load_balance_loss(weights, fm_ids, slice_mask=slice_mask)

    assert torch.allclose(loss, torch.zeros_like(loss), atol=1e-6)


def test_moco_router_z_loss_ignores_padded_slices():
    real_logits = torch.zeros(2, 2, 2, device=DEVICE)
    padded_logits = torch.full((2, 2, 2), 50.0, device=DEVICE)
    logits = torch.cat([real_logits, padded_logits], dim=1)
    slice_mask = torch.tensor(
        [[True, True, False, False], [True, True, False, False]],
        device=DEVICE,
    )

    masked = MoCo._router_z_loss(logits, slice_mask=slice_mask)
    expected = MoCo._router_z_loss(real_logits)

    assert torch.allclose(masked, expected, atol=1e-6)


def test_moco_router_confidence_weight_penalizes_entropy():
    model = _build_subset_moco(
        fm_pooling="router",
        num_fms=2,
        router_load_balance_weight=0.0,
        router_z_loss_weight=0.0,
        router_entropy_weight=1.0,
    ).eval()
    weights = torch.full((2, 3, 2), 0.5, device=DEVICE)
    entropy = torch.full((2, 3), torch.log(torch.tensor(2.0, device=DEVICE)), device=DEVICE)
    stats = {
        "fm_weights_local": weights,
        "fm_ids": torch.tensor([[0, 1], [0, 1]], dtype=torch.long, device=DEVICE),
        "router_logits": torch.zeros(2, 3, 2, device=DEVICE),
        "fm_weight_entropy": entropy,
    }

    out = model._router_aux_loss(torch.zeros((), device=DEVICE), [stats])

    assert out["loss"] > 0
    assert torch.allclose(out["loss"], out["loss_router_entropy"], atol=1e-6)


def test_moco_subset_router_backward():
    batch_size, k_sub, num_slices = 4, 2, 8
    model = _build_subset_moco(fm_pooling="router")
    model.train()
    x1, x2, dims, ids1, ids2 = _subset_views(batch_size, k_sub, num_slices)

    out = model(
        x1, x2,
        input_feature_dims_1=dims, input_feature_dims_2=dims,
        fm_ids_1=ids1, fm_ids_2=ids2, m=0.99,
    )
    out["loss"].backward()
    # Router parameters must receive gradients.
    router_grads = [
        p.grad is not None
        for n, p in model.base_encoder.named_parameters()
        if n.startswith("fm_router.") and p.requires_grad
    ]
    assert router_grads and all(router_grads)


def test_moco_subset_router_per_fm_id_adapter_forward_and_backward():
    """Subset-SSL router with per-FM adapters runs end-to-end and trains the adapters."""
    batch_size, k_sub, num_slices = 4, 2, 8
    # fm_input_dims ordered by global FM id: id0->64, id1->128, id2->128, id3->64.
    model = _build_subset_moco(
        fm_pooling="router",
        num_fms=4,
        per_fm_adapter_mode="per_fm_id",
        fm_input_dims=[64, 128, 128, 64],
    )
    model.train()
    x1, x2, dims, ids1, ids2 = _subset_views(batch_size, k_sub, num_slices)

    assert set(model.base_encoder.embed_fm.keys()) == {"0", "1", "2", "3"}

    out = model(
        x1, x2,
        input_feature_dims_1=dims, input_feature_dims_2=dims,
        fm_ids_1=ids1, fm_ids_2=ids2, m=0.99,
    )
    assert torch.isfinite(out["loss"]).all()
    out["loss"].backward()

    # Only the per-FM adapters present in the sampled subsets (ids 0, 1, 2) should train.
    used_grads = [
        model.base_encoder.embed_fm[str(i)].head[1].weight.grad is not None
        for i in (0, 1, 2)
    ]
    assert all(used_grads)


def test_moco_fm_fusion_requires_subset_ids():
    model = _build_subset_moco(fm_pooling="router").eval()
    x1 = torch.randn(4, 8, 64, device=DEVICE)
    x2 = torch.randn(4, 8, 64, device=DEVICE)
    dims = torch.full((4,), 64, dtype=torch.long, device=DEVICE)

    with pytest.raises(ValueError, match="requires subset FM mode"):
        model(x1, x2, input_feature_dims_1=dims, input_feature_dims_2=dims, m=0.99)


def test_moco_subset_packed_router_raises():
    model = _build_subset_moco(fm_pooling="router").eval()
    x1, x2, dims, ids1, ids2 = _subset_views(4, 2, 8)
    with pytest.raises(NotImplementedError):
        model(
            x1, x2,
            input_feature_dims_1=dims, input_feature_dims_2=dims,
            fm_ids_1=ids1, fm_ids_2=ids2, use_packed=True, m=0.99,
        )


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

