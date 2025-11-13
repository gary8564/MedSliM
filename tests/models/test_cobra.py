import torch
import pytest

from med_slim.model.sequence_encoder.cobra import Cobra


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_cobra_forward_pass():
    batch_size = 4
    num_slices = 16
    input_dim = 768
    embed_dim = 768
    contrast_dim = 256

    model = Cobra(
        embed_dim=embed_dim,
        contrast_dim=contrast_dim,
        input_dims=[512, 768, 1024, 1152, 1376],
        num_heads=4,
        layer=1,
        dropout=0.1,
        att_dim=128,
        d_state=64,
        mode="train",
    ).to(DEVICE).eval()

    x = torch.randn(batch_size, num_slices, input_dim, device=DEVICE)
    with torch.no_grad():
        y = model(x)
    assert isinstance(y, torch.Tensor)
    assert y.ndim == 2 and y.shape == (batch_size, contrast_dim)
    assert torch.isfinite(y).all()


def test_cobra_attention_shape():
    batch_size = 2
    num_slices = 10
    input_dim = 768

    model = Cobra(
        embed_dim=768,
        contrast_dim=128,
        input_dims=[768],
        num_heads=2,
        layer=1,
        dropout=0.0,
        att_dim=64,
        d_state=32,
        mode="train",
    ).to(DEVICE).eval()

    x = torch.randn(batch_size, num_slices, input_dim, device=DEVICE)
    with torch.no_grad():
        attn = model(x, get_attention=True)
    assert isinstance(attn, torch.Tensor)
    assert attn.shape == (batch_size, 1, num_slices)
    assert torch.isfinite(attn).all()


