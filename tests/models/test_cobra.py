import torch
import pytest
import os

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

def count_parameters(model):
    """Count total number of learnable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_attention_parameters(model):
    """Count parameters specifically in the attention module(s)"""
    if hasattr(model, 'attn'):
        if isinstance(model.attn, torch.nn.ModuleList):
            return sum(count_parameters(attn) for attn in model.attn)
        else:
            return count_parameters(model.attn)
    return 0

def get_pretrained_cobra_from_huggingface(config: dict):
    """Load pretrained COBRAII model from Hugging Face Hub."""
    from huggingface_hub import hf_hub_download
    download_path = hf_hub_download("KatherLab/COBRA", filename="cobraII.pth.tar", force_download=True)
    state_dict = torch.load(download_path, map_location="cpu", weights_only=False)
    model = Cobra(
        embed_dim=config['embed_dim'],
        contrast_dim=256,  # Not used in inference, but needed for initialization
        input_dims=config['input_dims'],
        num_heads=config['num_heads'],
        layer=config['layers'],
        dropout=config['dropout'],
        att_dim=config['att_dim'],
        d_state=config['d_state'],
        mode="inference"
    )
    if "state_dict" in list(state_dict.keys()):
        chkpt = state_dict["state_dict"]
        cobra_weights = {k.split("momentum_enc.")[-1]: v for k, v in chkpt.items() 
                        if "momentum_enc" in k and "momentum_enc.proj" not in k}
    else:
        cobra_weights = state_dict
    model.load_state_dict(cobra_weights, strict=False)
    return model

def test_cobra_parameter_count_vs_original():
    """Test total parameters match with the pretrained model from Hugging Face."""
    # COBRAII configuration (from https://github.com/KatherLab/COBRA/blob/main/cobra/utils/load_cobra.py)
    config = {
        'embed_dim': 768,
        'input_dims': [512, 1024, 1280, 1536],
        'num_heads': 4,
        'layers': 1,
        'dropout': 0.2,
        'att_dim': 256,
        'd_state': 128,
    }
    
    hf_model = get_pretrained_cobra_from_huggingface(config)
    
    # Create our implementation with same config 
    our_model = Cobra(
        embed_dim=config['embed_dim'],
        contrast_dim=256,  # Not in original, but needed for our implementation
        input_dims=config['input_dims'],
        num_heads=config['num_heads'],
        layer=config['layers'],
        dropout=config['dropout'],
        att_dim=config['att_dim'],
        d_state=config['d_state'],
        mode="inference",
    )
    
    # Count parameters for both models
    # Note: The projection layer is always created in the architecture (even in inference mode),
    # but its weights are not loaded from the checkpoint. So it exists with random initialization.
    hf_total = count_parameters(hf_model)
    hf_proj_params = count_parameters(hf_model.proj)
    hf_total_without_proj = hf_total - hf_proj_params
    hf_attn = count_attention_parameters(hf_model)
    
    our_total = count_parameters(our_model)
    our_proj_params = count_parameters(our_model.proj)
    our_total_without_proj = our_total - our_proj_params
    our_attn = count_attention_parameters(our_model)
    
    # Calculate expected parameter counts based on formula
    embed_dim = config['embed_dim']
    num_heads = config['num_heads']
    att_dim = config['att_dim']
    
    # Expected attention params: 2 * embed_dim * att_dim + 3 * num_heads * att_dim + num_heads
    expected_attn = 2 * embed_dim * att_dim + 3 * num_heads * att_dim + num_heads
    
    # Verify total parameters match between HF model and our implementation
    assert our_total == hf_total, \
        f"Total parameter count mismatch! HF model: {hf_total:,}, Ours: {our_total:,}"

    # Verify total parameters without projection layer match
    assert our_total_without_proj == hf_total_without_proj, \
        f"Total parameter count (without proj) mismatch! HF model: {hf_total_without_proj:,}, Ours: {our_total_without_proj:,}"
    
    # Verify attention parameters match between HF model and our implementation
    assert our_attn == hf_attn == expected_attn, \
        f"Attention parameter count mismatch! HF model: {hf_attn:,}, Ours: {our_attn:,}, Expected: {expected_attn:,}"
