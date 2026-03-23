import os
import pytest
import torch
from med_slim.model.slice_encoder import build_slice_encoder


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
B = 2


@pytest.mark.parametrize(
    "name,kwargs,input_shape,embed_dim",
    [
        # DINOv2 typical resolution 224x224
        ("dinov2", {"model_repo": "facebook/dinov2-base"}, (B, 1, 224, 224, 32), 768),
        # MedSigLIP default is 448x448
        ("medsiglip", {}, (B, 1, 448, 448, 32), 1152),
        # BiomedCLIP default is 224x224
        ("biomedclip", {}, (B, 1, 224, 224, 32), 512),
    ],
)
def test_slice_encoder_from_hf_models(name, kwargs, input_shape, embed_dim):
    model = build_slice_encoder(name, **kwargs)
    # Verify params are frozen by default
    assert all(not p.requires_grad for p in model.parameters())
    model.to(DEVICE).eval()
    x = torch.randn(*input_shape, device=DEVICE)
    with torch.no_grad():
        y = model(x)
    assert y is not None
    assert isinstance(y, torch.Tensor)
    assert y.ndim == 3 and y.shape == (B, 32, embed_dim)
    assert torch.isfinite(y).all()

@pytest.mark.skipif(
    not os.path.exists(
        os.environ.get(
            "ARK_CHECKPOINT",
            "/hpcwork/rwth1833/models/ark/Ark+_Nature/Ark6_swinLarge768_ep50.pth.tar",
        )
    ),
    reason="Ark checkpoint not on disk; set ARK_CHECKPOINT to run this test.",
)
def test_ark_slice_encoder():
    ckpt = os.environ.get(
        "ARK_CHECKPOINT",
        "/hpcwork/rwth1833/models/ark/Ark+_Nature/Ark6_swinLarge768_ep50.pth.tar",
    )
    model = build_slice_encoder("ark", checkpoint=ckpt, freeze=True)
    assert all(not p.requires_grad for p in model.parameters())
    model.to(DEVICE).eval()
    x = torch.randn(B, 1, 768, 768, 32, device=DEVICE)
    with torch.no_grad():
        y = model(x)
    assert isinstance(y, torch.Tensor)
    assert y.ndim == 3 and y.shape == (B, 32, 1376)
    assert torch.isfinite(y).all()
