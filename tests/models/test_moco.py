import torch
import pytest

from med_slim.model.ssl import MoCo


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


