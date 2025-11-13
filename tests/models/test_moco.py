import torch
import pytest

from med_slim.model.ssl import MoCo


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_moco_forward_loss():
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
        num_mamba_layers=1,
        T=0.2,
        dropout=0.0,
        att_dim=64,
        d_state=32,
    ).to(DEVICE).eval()

    x1 = torch.randn(batch_size, num_slices, input_dim, device=DEVICE)
    x2 = torch.randn(batch_size, num_slices, input_dim, device=DEVICE)
    with torch.no_grad():
        loss = model(x1, x2, input_feature_dims_1=None, input_feature_dims_2=None, m=0.99)
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # scalar tensor
    assert torch.isfinite(loss).all()


