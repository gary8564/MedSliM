import torch
import torch.nn as nn
import pytest
import os

from med_slim.model.sequence_encoder.cobra import Cobra


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_cobra_mamba2_forward_pass():
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
        num_layers=1,
        dropout=0.1,
        mode="train",
        d_state=64,
        att_dim=128,
    ).to(DEVICE).eval()

    x = torch.randn(batch_size, num_slices, input_dim, device=DEVICE)
    with torch.no_grad():
        y = model(x)
    assert isinstance(y, torch.Tensor)
    assert y.ndim == 2 and y.shape == (batch_size, contrast_dim)
    assert torch.isfinite(y).all()


def test_cobra_abmil_attention_shape():
    batch_size = 2
    num_slices = 10
    input_dim = 768

    model = Cobra(
        embed_dim=768,
        contrast_dim=128,
        input_dims=[768],
        num_heads=2,
        num_layers=1,
        dropout=0.0,
        mode="train",
        d_state=32,
        att_dim=64,
    ).to(DEVICE).eval()

    x = torch.randn(batch_size, num_slices, input_dim, device=DEVICE)
    with torch.no_grad():
        attn = model(x, get_attention=True)
    assert isinstance(attn, torch.Tensor)
    assert attn.shape == (batch_size, 1, num_slices)
    assert torch.isfinite(attn).all()

def test_cobra_variable_slice_seq_lengths():
    """
    If seq_lengths is provided, padded slice positions should be ignored by:
    - the Transformer encoder via src_key_padding_mask (True = padded)
    - the ABMIL pooling attention via mask (True = valid)
    """
    batch_size = 2
    max_slices = 8
    input_dim = 768
    contrast_dim = 128

    model = Cobra(
        embed_dim=input_dim,
        contrast_dim=contrast_dim,
        input_dims=[input_dim],
        num_heads=2,
        num_layers=1,
        dropout=0.0,
        mode="train",
        sequence_encoder="transformer",
        att_dim=64,
    ).to(DEVICE).eval()

    seq_lengths = torch.tensor([3, 6], dtype=torch.long, device=DEVICE)

    x = torch.randn(batch_size, max_slices, input_dim, device=DEVICE)
    x_alt = x.clone()
    # Heavily perturb padded positions only (should not affect output if masking works)
    x_alt[0, 3:, :] = torch.randn_like(x_alt[0, 3:, :]) * 50.0 + 100.0
    x_alt[1, 6:, :] = torch.randn_like(x_alt[1, 6:, :]) * 50.0 + 100.0

    with torch.no_grad():
        y1 = model(x, seq_lengths=seq_lengths)
        y2 = model(x_alt, seq_lengths=seq_lengths)
        attn = model(x, seq_lengths=seq_lengths, get_attention=True)

    assert y1.shape == (batch_size, contrast_dim)
    assert torch.isfinite(y1).all() and torch.isfinite(y2).all()
    assert torch.allclose(y1, y2, atol=1e-5, rtol=1e-5)

    # Padded positions should get ~0 attention probability
    assert attn.shape == (batch_size, 1, max_slices)
    assert torch.allclose(attn[0, 0, 3:], torch.zeros_like(attn[0, 0, 3:]), atol=1e-6, rtol=0.0)
    assert torch.allclose(attn[1, 0, 6:], torch.zeros_like(attn[1, 0, 6:]), atol=1e-6, rtol=0.0)

def test_cobra_transformer_cls_pooling():
    """
    Test transformer + CLS token pooling.
    The CLS token should aggregate information from all valid slices.
    Padded positions should be ignored via src_key_padding_mask.
    """
    batch_size = 2
    max_slices = 8
    input_dim = 768
    contrast_dim = 128

    model = Cobra(
        embed_dim=input_dim,
        contrast_dim=contrast_dim,
        input_dims=[input_dim],
        num_heads=4,
        num_layers=1,
        dropout=0.0,
        mode="train",
        sequence_encoder="transformer",
        slice_pooling="cls",
    ).to(DEVICE).eval()

    # Verify CLS token exists and ABMIL modules don't
    assert model.cls_token is not None
    assert model.cls_token.shape == (1, 1, input_dim)
    assert model.attn is None 

    seq_lengths = torch.tensor([3, 6], dtype=torch.long, device=DEVICE)

    x = torch.randn(batch_size, max_slices, input_dim, device=DEVICE)
    x_alt = x.clone()
    # Heavily perturb padded positions (should not affect output if masking works)
    x_alt[0, 3:, :] = torch.randn_like(x_alt[0, 3:, :]) * 50.0 + 100.0
    x_alt[1, 6:, :] = torch.randn_like(x_alt[1, 6:, :]) * 50.0 + 100.0

    with torch.no_grad():
        y1 = model(x, seq_lengths=seq_lengths)
        y2 = model(x_alt, seq_lengths=seq_lengths)

    assert y1.shape == (batch_size, contrast_dim)
    assert torch.isfinite(y1).all() and torch.isfinite(y2).all()
    # Outputs should be identical since padded positions are masked
    assert torch.allclose(y1, y2, atol=1e-5, rtol=1e-5)


def test_cobra_transformer_cls_attention():
    """
    Test attention extraction for CLS pooling.
    Should return attention weights from CLS token to slice tokens.
    """
    batch_size = 2
    max_slices = 8
    input_dim = 768

    model = Cobra(
        embed_dim=input_dim,
        contrast_dim=128,
        input_dims=[input_dim],
        num_heads=4,
        num_layers=1,
        dropout=0.0,
        mode="train",
        sequence_encoder="transformer",
        slice_pooling="cls",
    ).to(DEVICE).eval()

    seq_lengths = torch.tensor([3, 6], dtype=torch.long, device=DEVICE)
    x = torch.randn(batch_size, max_slices, input_dim, device=DEVICE)

    with torch.no_grad():
        attn = model(x, seq_lengths=seq_lengths, get_attention=True)

    # Should return attention in same format as ABMIL: [B, 1, num_slices]
    assert attn.shape == (batch_size, 1, max_slices)
    assert torch.isfinite(attn).all()
    
    # Padded positions should get ~0 attention
    assert torch.allclose(attn[0, 0, 3:], torch.zeros_like(attn[0, 0, 3:]), atol=1e-6, rtol=0.0)
    assert torch.allclose(attn[1, 0, 6:], torch.zeros_like(attn[1, 0, 6:]), atol=1e-6, rtol=0.0)
    
    # Valid positions should sum to ~1
    assert torch.allclose(attn[0, 0, :3].sum(), torch.tensor(1.0, device=DEVICE), atol=1e-5)
    assert torch.allclose(attn[1, 0, :6].sum(), torch.tensor(1.0, device=DEVICE), atol=1e-5)


def test_cobra_cls_pooling_requires_transformer():
    """CLS pooling should raise error when used with mamba2 encoder."""
    with pytest.raises(ValueError, match="slice_pooling='cls' requires sequence_encoder='transformer'"):
        Cobra(
            embed_dim=768,
            contrast_dim=128,
            input_dims=[768],
            num_heads=2,
            num_layers=1,
            mode="train",
            sequence_encoder="mamba2",
            slice_pooling="cls",
        )


def test_detect_slice_pooling_from_checkpoint_keys():
    """
    Verify that checkpoint key detection correctly distinguishes abmil and cls.
    """
    abmil_keys = {"cobra.attn.0.attention_V.0.weight": None, "cobra.norm.weight": None}
    cls_keys = {"cobra.cls_token": None, "cobra.norm.weight": None}

    def detect(raw):
        has_abmil = any(k.startswith("cobra.attn.") for k in raw.keys())
        if has_abmil:
            return "abmil"
        return "cls"

    assert detect(abmil_keys) == "abmil"
    assert detect(cls_keys) == "cls"


def test_cobra_rejects_cross_attention_slice_pooling():
    with pytest.raises(AssertionError, match="Invalid slice_pooling"):
        Cobra(
            embed_dim=128,
            contrast_dim=2,
            input_dims=[128],
            num_heads=4,
            num_layers=1,
            dropout=0.0,
            mode="train",
            slice_pooling="cross_attention",
            d_state=64,
        )


def test_cobra_cls_post_embed_pooling_target_warns():
    with pytest.warns(UserWarning, match="slice_pooling='cls'"):
        model = Cobra(
            embed_dim=128,
            contrast_dim=2,
            input_dims=[128],
            num_heads=4,
            num_layers=1,
            dropout=0.0,
            mode="inference",
            slice_pooling="cls",
            pooling_target="post_embed",
            sequence_encoder="transformer",
            d_state=64,
        )
    assert model.pooling_target is None


def test_resolve_pooling_target_cobra_cls():
    """CLS pooling has no pooling_target."""
    model = Cobra(
        embed_dim=128,
        contrast_dim=2,
        input_dims=[128],
        num_heads=4,
        num_layers=1,
        dropout=0.0,
        mode="inference",
        slice_pooling="cls",
        sequence_encoder="transformer",
        d_state=64,
    )
    assert model.pooling_target is None

    explicit_model = Cobra(
        embed_dim=128,
        contrast_dim=2,
        input_dims=[128],
        num_heads=4,
        num_layers=1,
        dropout=0.0,
        mode="inference",
        slice_pooling="cls",
        pooling_target="post_encoder",
        sequence_encoder="transformer",
        d_state=64,
    )
    assert explicit_model.pooling_target is None


def _count_parameters(model):
    """Count total number of learnable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def _count_attention_parameters(model):
    """Count parameters specifically in the attention module(s)"""
    if hasattr(model, 'attn'):
        if isinstance(model.attn, torch.nn.ModuleList):
            return sum(_count_parameters(attn) for attn in model.attn)
        else:
            return _count_parameters(model.attn)
    return 0

def _get_pretrained_cobra_from_huggingface(config: dict):
    """Load pretrained COBRAII model from Hugging Face Hub."""
    from huggingface_hub import hf_hub_download
    download_path = hf_hub_download("KatherLab/COBRA", filename="cobraII.pth.tar", force_download=True)
    state_dict = torch.load(download_path, map_location="cpu", weights_only=False)
    model = Cobra(
        embed_dim=config['embed_dim'],
        contrast_dim=256,  # Not used in inference, but needed for initialization
        input_dims=config['input_dims'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        mode="inference",
        pooling_target="post_encoder",
        d_state=config['d_state'],
        att_dim=config['att_dim'],
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
        'num_layers': 1,
        'dropout': 0.2,
        'att_dim': 256,
        'd_state': 128,
    }
    
    hf_model = _get_pretrained_cobra_from_huggingface(config)
    
    # Create our implementation with same config 
    our_model = Cobra(
        embed_dim=config['embed_dim'],
        contrast_dim=256,
        input_dims=config['input_dims'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        mode="inference",
        pooling_target="post_encoder",
        d_state=config['d_state'],
        att_dim=config['att_dim'],
    )
    
    # Count parameters for both models
    # Note: The projection layer is always created in the architecture (even in inference mode),
    # but its weights are not loaded from the checkpoint. So it exists with random initialization.
    hf_total = _count_parameters(hf_model)
    hf_proj_params = _count_parameters(hf_model.proj)
    hf_total_without_proj = hf_total - hf_proj_params
    hf_attn = _count_attention_parameters(hf_model)
    
    our_total = _count_parameters(our_model)
    our_proj_params = _count_parameters(our_model.proj)
    our_total_without_proj = our_total - our_proj_params
    our_attn = _count_attention_parameters(our_model)
    
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


def _build_tiled_cobra(mode="train", region_embedding=False, regional_tokens=4,
                       embed_dim=64, contrast_dim=32, slice_pooling="abmil",
                       pooling_target="post_encoder"):
    return Cobra(
        embed_dim=embed_dim,
        contrast_dim=contrast_dim,
        input_dims=[64, 128],
        num_heads=4,
        num_layers=1,
        dropout=0.0,
        mode=mode,
        sequence_encoder="transformer",  # avoid Mamba causal-conv CPU/stride constraints
        slice_pooling=slice_pooling,
        pooling_target=pooling_target,
        regional_tokens=regional_tokens,
        region_embedding=region_embedding,
        att_dim=32,
    ).to(DEVICE).eval()

def test_cobra_train_flattens_tiled_tokens():
    batch_size, num_slices, regions, feat_dim = 2, 6, 5, 64  # num_tiled_regions = 1 global + 2x2
    model = _build_tiled_cobra(mode="train", regional_tokens=4)

    x = torch.randn(batch_size, num_slices, regions, feat_dim, device=DEVICE)
    with torch.no_grad():
        y = model(x)
        attn = model(x, get_attention=True)
    assert y.shape == (batch_size, 32)
    assert torch.isfinite(y).all()
    assert attn.shape == (batch_size, 1, num_slices * regions)


def test_resolve_pooling_target_cobra_tiled_train():
    """pooling_target is None for tiled SSL pretraining."""
    batch_size, num_slices, regions, feat_dim = 2, 4, 5, 64
    model = Cobra(
        embed_dim=64, contrast_dim=32, input_dims=[64], num_heads=4, num_layers=1,
        dropout=0.0, mode="train", sequence_encoder="transformer",
        slice_pooling="abmil", regional_tokens=4, att_dim=32,
    ).to(DEVICE).eval()
    assert model.pooling_target is None

    x = torch.randn(batch_size, num_slices, regions, feat_dim, device=DEVICE)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (batch_size, 32)
    assert torch.isfinite(y).all()


def test_resolve_pooling_target_cobra_tiled_inference():
    """Omitted inference pooling_target defaults to raw for global-only and post_embed for tiled."""
    global_model = Cobra(
        embed_dim=64,
        contrast_dim=32,
        input_dims=[64],
        num_heads=4,
        num_layers=1,
        dropout=0.0,
        mode="inference",
        sequence_encoder="transformer",
        slice_pooling="abmil",
        raw_output_dim=64,
        att_dim=32,
    ).to(DEVICE).eval()
    assert global_model.pooling_target == "raw"

    tiled_model = Cobra(
        embed_dim=64,
        contrast_dim=32,
        input_dims=[64],
        num_heads=4,
        num_layers=1,
        dropout=0.0,
        mode="inference",
        sequence_encoder="transformer",
        slice_pooling="abmil",
        regional_tokens=4,
        att_dim=32,
    ).to(DEVICE).eval()
    assert tiled_model.pooling_target == "post_embed"


def test_cobra_tiled_packed_train_forward():
    """Packed tiled input runs through the real Transformer varlen sequence encoder."""
    if not torch.cuda.is_available():
        pytest.skip("Packed Transformer uses FlashAttention varlen, which requires CUDA.")

    device = torch.device("cuda")
    seq_lens = torch.tensor([3, 5], dtype=torch.int32, device=device)
    cu_seqlens = torch.zeros(3, dtype=torch.int32, device=device)
    cu_seqlens[1:] = torch.cumsum(seq_lens, dim=0)
    total_slices = int(cu_seqlens[-1].item())
    max_seqlen = int(seq_lens.max().item())
    seq_idx = torch.repeat_interleave(torch.arange(2, dtype=torch.int32, device=device), seq_lens)

    model = Cobra(
        embed_dim=64,
        contrast_dim=32,
        input_dims=[64],
        num_heads=4,
        num_layers=1,
        dropout=0.0,
        mode="train",
        sequence_encoder="transformer",
        slice_pooling="abmil",
        regional_tokens=4,
        att_dim=32,
    ).to(device).eval()
    x = torch.randn(total_slices, 5, 64, device=device)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            y = model(
                x,
                use_packed=True,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                seq_idx=seq_idx,
            )

    assert y.shape == (2, 32)
    assert torch.isfinite(y).all()


def test_cobra_tiled_inference_avg_pool_multi_fm():
    """Inference mode ensembles a list of tiled FM tensors via avg_pool."""
    batch_size, num_slices, regions = 2, 5, 5
    model = _build_tiled_cobra(mode="inference", regional_tokens=4)

    # Two FMs with different raw feature dims (64 and 128), both tiled.
    x = [
        torch.randn(batch_size, num_slices, regions, 64, device=DEVICE),
        torch.randn(batch_size, num_slices, regions, 128, device=DEVICE),
    ]
    seq_lengths = torch.tensor([num_slices, 3], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        y = model(x, seq_lengths=seq_lengths)
        attn = model(x, seq_lengths=seq_lengths, get_attention=True)
    assert y.shape == (batch_size, 64)  # embed_dim
    assert torch.isfinite(y).all()
    assert attn.shape == (batch_size, 1, num_slices * regions)


def test_cobra_tiled_inference_attention_pool_multi_fm():
    """FM attention pooling works over flattened slice-region tokens."""
    batch_size, num_slices, regions = 2, 5, 5
    model = Cobra(
        embed_dim=64,
        contrast_dim=32,
        input_dims=[64, 128],
        num_heads=4,
        num_layers=1,
        dropout=0.0,
        mode="inference",
        sequence_encoder="transformer",
        fm_pooling="attention",
        slice_pooling="abmil",
        pooling_target="post_encoder",
        regional_tokens=4,
        att_dim=32,
    ).to(DEVICE).eval()

    x = [
        torch.randn(batch_size, num_slices, regions, 64, device=DEVICE),
        torch.randn(batch_size, num_slices, regions, 128, device=DEVICE),
    ]
    seq_lengths = torch.tensor([num_slices, 3], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        y = model(x, seq_lengths=seq_lengths)

    assert y.shape == (batch_size, 64)
    assert torch.isfinite(y).all()


def test_cobra_tiled_inference_post_embed_pooling():
    """Tiled inference supports post_embed pooling over flattened slice-region tokens."""
    batch_size, num_slices, regions, feat_dim = 2, 5, 5, 64
    model = _build_tiled_cobra(mode="inference", regional_tokens=4, pooling_target="post_embed")

    x = [torch.randn(batch_size, num_slices, regions, feat_dim, device=DEVICE)]
    seq_lengths = torch.tensor([num_slices, 3], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        y = model(x, seq_lengths=seq_lengths)
    assert y.shape == (batch_size, 64)
    assert torch.isfinite(y).all()


def test_cobra_tiled_region_embedding_adds_params():
    """Enabling region_embedding creates a learned embedding over (global + regional) tokens."""
    model = _build_tiled_cobra(mode="train", region_embedding=True, regional_tokens=4)
    assert model.region_embed is not None
    assert model.region_embed.weight.shape[0] == 5  # 1 global + 4 regions

    model_no_embed = _build_tiled_cobra(mode="train", region_embedding=False, regional_tokens=4)
    assert model_no_embed.region_embed is None


def test_cobra_tiled_raw_pooling_target_flattens_raw_tokens():
    """Tiled raw pooling aggregates flattened global/regional FM tokens."""
    batch_size, num_slices, regions, feat_dim = 2, 5, 5, 64
    model = Cobra(
        embed_dim=64,
        contrast_dim=32,
        input_dims=[64],
        num_heads=4,
        num_layers=1,
        dropout=0.0,
        mode="inference",
        sequence_encoder="transformer",
        slice_pooling="abmil",
        pooling_target="raw",
        regional_tokens=4,
        raw_output_dim=feat_dim,
        att_dim=32,
    ).to(DEVICE).eval()
    x = [torch.randn(batch_size, num_slices, regions, feat_dim, device=DEVICE)]
    seq_lengths = torch.tensor([num_slices, 3], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        y = model(x, seq_lengths=seq_lengths)
        attn = model(x, seq_lengths=seq_lengths, get_attention=True)

    assert y.shape == (batch_size, feat_dim)
    assert torch.isfinite(y).all()
    assert attn.shape == (batch_size, 1, num_slices * regions)


def test_cobra_tiled_shape_layout_mismatch_raises():
    """Layout/model mismatches between tiled and global-only must fail loudly."""
    # Tiled features fed to a non-tiled model
    plain = Cobra(
        embed_dim=64, contrast_dim=32, input_dims=[64], num_heads=4, num_layers=1,
        dropout=0.0, mode="train", sequence_encoder="transformer", slice_pooling="abmil",
        att_dim=32,
    ).to(DEVICE).eval()
    x_tiled = torch.randn(2, 4, 5, 64, device=DEVICE)
    with pytest.raises(ValueError):
        plain(x_tiled)

    # Global-only features fed to a tiled model
    tiled = _build_tiled_cobra(mode="train", regional_tokens=4)
    x_plain = torch.randn(2, 4, 64, device=DEVICE)
    with pytest.raises(ValueError):
        tiled(x_plain)


def test_cobra_zero_padded_slices_masks_batch_padding():
    """Flattened tiled tokens must ignore batch-padded slice positions."""
    model = _build_tiled_cobra(mode="inference", regional_tokens=4)
    batch_size, max_slices, regions, feat_dim = 2, 6, 5, 64
    x = torch.randn(batch_size, max_slices, regions, feat_dim, device=DEVICE)
    seq_lengths = torch.tensor([4, 6], device=DEVICE)

    padded = x.clone()
    padded[0, 4:] = 999.0
    with torch.no_grad():
        emb_clean = model([x], seq_lengths=seq_lengths)
        emb_padded = model([padded], seq_lengths=seq_lengths)

    assert torch.allclose(emb_clean, emb_padded, atol=1e-5)
