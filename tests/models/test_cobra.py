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


# Subset SSL multi-FM fusion (router / avg_pool)
NUM_FMS = 4
SUBSET_DIMS = [64, 128]  # raw feature dims of the two FMs

def _build_subset_cobra(fm_pooling="router", num_fms=NUM_FMS,
                        embed_dim=64, contrast_dim=32, **kwargs):
    return Cobra(
        embed_dim=embed_dim,
        contrast_dim=contrast_dim,
        input_dims=[64, 128],
        num_heads=4,
        num_layers=1,
        dropout=0.0,
        mode="train",
        sequence_encoder="transformer",
        slice_pooling="abmil",
        fm_pooling=fm_pooling,
        num_fms=num_fms,
        att_dim=32,
        **kwargs,
    ).to(DEVICE).eval()


def _generate_subset_inputs(batch_size, k_sub, num_slices, max_feature_dim=128, num_regions=None):
    """Return (x, input_feature_dims, fm_ids) for a subset-mode forward pass."""
    dims = [SUBSET_DIMS[i % len(SUBSET_DIMS)] for i in range(k_sub)]
    if num_regions is None:
        x = torch.randn(batch_size, k_sub, num_slices, max_feature_dim, device=DEVICE)
    else:
        x = torch.randn(batch_size, k_sub, num_slices, num_regions, max_feature_dim, device=DEVICE)
    input_feature_dims = torch.tensor([dims] * batch_size, dtype=torch.long, device=DEVICE)
    fm_ids = torch.tensor([list(range(k_sub))] * batch_size, dtype=torch.long, device=DEVICE)
    return x, input_feature_dims, fm_ids


def test_cobra_subset_router_global():
    batch_size, k_sub, num_slices = 2, 2, 6
    model = _build_subset_cobra(fm_pooling="router")
    assert model.fm_router is not None

    x, dims, fm_ids = _generate_subset_inputs(batch_size, k_sub, num_slices)
    with torch.no_grad():
        y = model(x, input_feature_dims=dims, fm_ids=fm_ids)

    assert y.shape == (batch_size, 32)
    assert torch.isfinite(y).all()
    stats = model._last_fm_stats
    assert stats is not None
    assert stats["router_logits"] is not None
    assert stats["fm_weights_local"].shape == (batch_size, num_slices, k_sub)
    sums = stats["fm_weights_local"].sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-4)


def test_cobra_train_flattened_regional_tokens():
    batch_size, num_slices, regional_tokens = 2, 5, 4
    num_regions = 1 + regional_tokens
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
        regional_tokens=regional_tokens,
        att_dim=32,
    ).to(DEVICE).eval()
    x = torch.randn(batch_size, num_slices, num_regions, 64, device=DEVICE)
    seq_lengths = torch.tensor([num_slices, 3], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        h = model(x, seq_lengths=seq_lengths, return_slice_embeddings=True)
        attn = model(x, seq_lengths=seq_lengths, get_attention=True)

    assert h.shape == (batch_size, num_slices * num_regions, 64)
    assert attn.shape == (batch_size, 1, num_slices * num_regions)
    assert torch.allclose(
        attn[1, 0, 3 * num_regions:],
        torch.zeros_like(attn[1, 0, 3 * num_regions:]),
        atol=1e-6,
        rtol=0.0,
    )


def test_cobra_subset_router_flattened_regional_tokens():
    batch_size, k_sub, num_slices, regional_tokens = 2, 2, 5, 4
    num_regions = 1 + regional_tokens
    model = _build_subset_cobra(fm_pooling="router", regional_tokens=regional_tokens)

    x, dims, fm_ids = _generate_subset_inputs(
        batch_size, k_sub, num_slices, num_regions=num_regions
    )
    seq_lengths = torch.tensor([num_slices, 3], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        y = model(x, input_feature_dims=dims, fm_ids=fm_ids, seq_lengths=seq_lengths)

    assert y.shape == (batch_size, 32)
    stats = model._last_fm_stats
    assert stats["fm_weights_local"].shape == (
        batch_size,
        num_slices * num_regions,
        k_sub,
    )
    assert torch.equal(
        stats["slice_mask"],
        model._build_mask(seq_lengths * num_regions, num_slices * num_regions),
    )


def test_cobra_variable_k_router_transfer():
    """The same router runs with K_sub=2 (train-like) and K_total=4 (inference-like)."""
    batch_size, num_slices = 2, 5
    model = _build_subset_cobra(fm_pooling="router", num_fms=4)

    with torch.no_grad():
        x2, dims2, ids2 = _generate_subset_inputs(batch_size, 2, num_slices)
        y2 = model(x2, input_feature_dims=dims2, fm_ids=ids2)
        x4, dims4, ids4 = _generate_subset_inputs(batch_size, 4, num_slices)
        y4 = model(x4, input_feature_dims=dims4, fm_ids=ids4)

    assert y2.shape == (batch_size, 32)
    assert y4.shape == (batch_size, 32)
    assert model._last_fm_stats["fm_weights_local"].shape == (batch_size, num_slices, 4)


def test_cobra_router_inference_uses_global_fm_ids():
    """Router inference must preserve checkpoint-global FM IDs, not local list positions."""
    batch_size, num_slices, feat_dim = 2, 5, 64
    model = Cobra(
        embed_dim=64,
        contrast_dim=32,
        input_dims=[feat_dim],
        num_heads=4,
        num_layers=1,
        dropout=0.0,
        mode="inference",
        sequence_encoder="transformer",
        slice_pooling="abmil",
        fm_pooling="router",
        num_fms=4,
        pooling_target="post_encoder",
        att_dim=32,
    ).to(DEVICE).eval()
    x = [
        torch.randn(batch_size, num_slices, feat_dim, device=DEVICE),
        torch.randn(batch_size, num_slices, feat_dim, device=DEVICE),
    ]
    fm_ids = torch.tensor([2, 3], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        y = model(x, fm_ids=fm_ids)

    assert y.shape == (batch_size, 64)
    expected_ids = fm_ids.unsqueeze(0).expand(batch_size, -1)
    assert torch.equal(model._last_fm_stats["fm_ids"], expected_ids)


def test_cobra_router_requires_num_fms():
    with pytest.raises(ValueError, match="num_fms"):
        Cobra(
            embed_dim=64, contrast_dim=32, input_dims=[64], num_heads=4, num_layers=1,
            mode="train", sequence_encoder="transformer", slice_pooling="abmil",
            fm_pooling="router", att_dim=32,
        )


def test_cobra_subset_packed_router_raises():
    model = _build_subset_cobra(fm_pooling="router")
    x, dims, fm_ids = _generate_subset_inputs(2, 2, 4)
    with pytest.raises(NotImplementedError, match="Packed subset"):
        model(x, input_feature_dims=dims, fm_ids=fm_ids, use_packed=True)


def test_cobra_inference_packed_raises():
    """Packed inference is unsupported; use padded [B, D, F] inputs plus seq_lengths."""
    batch_size, num_slices, feat_dim = 2, 5, 64
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
        pooling_target="post_encoder",
        att_dim=32,
    ).to(DEVICE).eval()

    x = torch.randn(num_slices + 3, feat_dim, device=DEVICE)  # packed [total_slices, F]
    cu_seqlens = torch.tensor([0, num_slices, num_slices + 3], dtype=torch.int32, device=DEVICE)
    seq_idx = torch.repeat_interleave(
        torch.arange(batch_size, dtype=torch.int32, device=DEVICE),
        torch.tensor([num_slices, 3], dtype=torch.int32, device=DEVICE),
    )
    with pytest.raises(NotImplementedError, match="Packed inference"):
        model(
            [x],
            use_packed=True,
            cu_seqlens=cu_seqlens,
            max_seqlen=num_slices,
            seq_idx=seq_idx,
        )


# Single-FM inference mode
def test_cobra_single_fm_inference_tensor_matches_list_global():
    """Single-FM inference accepts a bare tensor and matches a one-element list."""
    batch_size, num_slices, feat_dim = 2, 5, 64
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
        pooling_target="post_encoder",
        att_dim=32,
    ).to(DEVICE).eval()

    x = torch.randn(batch_size, num_slices, feat_dim, device=DEVICE)
    seq_lengths = torch.tensor([num_slices, 3], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        y_tensor = model(x, seq_lengths=seq_lengths)
        y_list = model([x], seq_lengths=seq_lengths)

    assert y_tensor.shape == (batch_size, 64)
    assert torch.allclose(y_tensor, y_list, atol=1e-6)


# Per-FM projection adapters (per_fm_adapter_mode='per_fm_id')
def _build_per_fm_cobra(mode="train", fm_input_dims=(64, 128, 128), num_fms=3):
    return Cobra(
        embed_dim=64,
        contrast_dim=32,
        input_dims=[64, 128],
        num_heads=4,
        num_layers=1,
        dropout=0.0,
        mode=mode,
        sequence_encoder="transformer",
        slice_pooling="abmil",
        pooling_target="post_encoder" if mode == "inference" else None,
        fm_pooling="router",
        num_fms=num_fms,
        per_fm_adapter_mode="per_fm_id",
        fm_input_dims=list(fm_input_dims),
        att_dim=32,
    ).to(DEVICE).eval()


def test_cobra_builds_one_embed_per_fm():
    model = _build_per_fm_cobra(fm_input_dims=(64, 128, 128), num_fms=3)
    assert model.per_fm_adapter_mode == "per_fm_id"
    assert set(model.embed_fm.keys()) == {"0", "1", "2"}
    # Each adapter projects from its FM's true input dim (head[1] is Linear(dim, embed_dim)).
    assert model.embed_fm["0"].head[1].in_features == 64
    assert model.embed_fm["1"].head[1].in_features == 128
    assert model.embed_fm["2"].head[1].in_features == 128
    # Per-FM mode does not instantiate unused dim-keyed adapters/checkpoint params.
    assert model.embed is None
    assert not any(k.startswith("embed.") for k in model.state_dict())
    assert any(k.startswith("embed_fm.") for k in model.state_dict())


def test_cobra_per_fm_id_requires_fm_input_dims():
    with pytest.raises(ValueError, match="requires fm_input_dims"):
        Cobra(
            embed_dim=64,
            contrast_dim=32,
            input_dims=[64, 128],
            num_heads=4,
            num_layers=1,
            mode="train",
            sequence_encoder="transformer",
            fm_pooling="router",
            num_fms=3,
            per_fm_adapter_mode="per_fm_id",
            fm_input_dims=None,
            att_dim=32,
        )


def test_cobra_per_fm_id_length_mismatch_raises():
    with pytest.raises(ValueError, match="must equal num_fms"):
        Cobra(
            embed_dim=64,
            contrast_dim=32,
            input_dims=[64, 128],
            num_heads=4,
            num_layers=1,
            mode="train",
            sequence_encoder="transformer",
            fm_pooling="router",
            num_fms=3,
            per_fm_adapter_mode="per_fm_id",
            fm_input_dims=[64, 128],
            att_dim=32,
        )


def test_cobra_per_fm_id_pretrain_subset_embed_selects_adapter_by_fm_id():
    """_embed_subset_fm_set must route each (sample, FM) through its FM-id adapter."""
    model = _build_per_fm_cobra(fm_input_dims=(64, 128, 128), num_fms=3)
    batch_size, k_sub, num_slices, max_dim = 2, 2, 4, 128
    x = torch.randn(batch_size, k_sub, num_slices, max_dim, device=DEVICE)
    # Subset position 0 -> FM id 0 (dim 64), position 1 -> FM id 2 (dim 128).
    fm_ids = torch.tensor([[0, 2]] * batch_size, dtype=torch.long, device=DEVICE)
    dims = torch.tensor([[64, 128]] * batch_size, dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        out = model._embed_subset_fm_set(x, dims, fm_ids=fm_ids)  # [K, B, D, E]
        # Reference: apply each FM's adapter explicitly.
        ref0 = model.embed_fm["0"](x[:, 0, :, :64])
        ref1 = model.embed_fm["2"](x[:, 1, :, :128])

    assert out.shape == (k_sub, batch_size, num_slices, 64)
    assert torch.allclose(out[0], ref0, atol=1e-6)
    assert torch.allclose(out[1], ref1, atol=1e-6)


def test_cobra_per_fm_id_pretrain_subset_embed_requires_fm_ids():
    model = _build_per_fm_cobra(fm_input_dims=(64, 128, 128), num_fms=3)
    x = torch.randn(2, 2, 4, 128, device=DEVICE)
    dims = torch.tensor([[64, 128]] * 2, dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        with pytest.raises(ValueError, match="requires fm_ids"):
            model._embed_subset_fm_set(x, dims, fm_ids=None)


# Explicit raw aggregation FM (pooling_target='raw')
def test_cobra_raw_pooling_requires_raw_aggregation_index():
    with pytest.raises(ValueError, match="raw_aggregation_index is required"):
        Cobra(
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
            raw_output_dim=64,
            att_dim=32,
        )


def test_cobra_raw_pooling_single_fm_matches_manual_bmm():
    """K=1: raw pooling aggregates the sole FM's raw features regardless of raw_aggregation_index."""
    batch_size, num_slices, feat_dim = 2, 6, 64
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
        raw_output_dim=64,
        raw_aggregation_index=0,
        att_dim=32,
    ).to(DEVICE).eval()

    x = torch.randn(batch_size, num_slices, feat_dim, device=DEVICE)
    with torch.no_grad():
        y_list = model([x])
        y_tensor = model(x)  # bare tensor is normalized to a length-1 list
        A = model([x], get_attention=True)
        expected = torch.bmm(A, x).squeeze(1)

    assert y_list.shape == (batch_size, feat_dim)
    assert torch.allclose(y_list, expected, atol=1e-6)
    assert torch.allclose(y_tensor, expected, atol=1e-6)


def _build_multi_fm_raw_cobra(fm_pooling="avg_pool", raw_aggregation_index=0, num_fms=None, raw_output_dim=None, **kwargs):
    dims = [64, 96]
    if raw_output_dim is None:
        raw_output_dim = dims[raw_aggregation_index] if 0 <= raw_aggregation_index < len(dims) else 64
    return Cobra(
        embed_dim=64,
        contrast_dim=32,
        input_dims=dims,
        num_heads=4,
        num_layers=1,
        dropout=0.0,
        mode="inference",
        sequence_encoder="transformer",
        slice_pooling="abmil",
        fm_pooling=fm_pooling,
        num_fms=num_fms,
        pooling_target="raw",
        raw_output_dim=raw_output_dim,
        raw_aggregation_index=raw_aggregation_index,
        att_dim=32,
        **kwargs,
    ).to(DEVICE).eval()


def test_cobra_raw_pooling_multi_fm_avg_pool_selects_named_index():
    """K=2 heterogeneous-dim FMs, avg_pool fusion: raw pooling aggregates only the selected FM."""
    batch_size, num_slices = 2, 5
    x = [
        torch.randn(batch_size, num_slices, 64, device=DEVICE),
        torch.randn(batch_size, num_slices, 96, device=DEVICE),
    ]

    model0 = _build_multi_fm_raw_cobra(raw_aggregation_index=0)
    with torch.no_grad():
        y0 = model0(x)
        A0 = model0(x, get_attention=True)
    assert y0.shape == (batch_size, 64)
    assert torch.allclose(y0, torch.bmm(A0, x[0]).squeeze(1), atol=1e-6)

    model1 = _build_multi_fm_raw_cobra(raw_aggregation_index=1)
    model1.load_state_dict(model0.state_dict())  # share weights for a like-for-like attention map
    with torch.no_grad():
        y1 = model1(x)
        A1 = model1(x, get_attention=True)
    assert y1.shape == (batch_size, 96)
    assert torch.allclose(y1, torch.bmm(A1, x[1]).squeeze(1), atol=1e-6)

    # Shared multi-FM attention (from the fused, sequence-encoded representation) is
    # identical regardless of which FM's raw features it is ultimately applied to.
    assert torch.allclose(A0, A1, atol=1e-6)


def test_cobra_raw_pooling_multi_fm_router_allowed():
    """Router SSL checkpoints may now use pooling_target='raw' via an explicit raw_aggregation_index."""
    batch_size, num_slices = 2, 5
    model = _build_multi_fm_raw_cobra(fm_pooling="router", raw_aggregation_index=1, num_fms=4)
    assert model.fm_router is not None

    x = [
        torch.randn(batch_size, num_slices, 64, device=DEVICE),
        torch.randn(batch_size, num_slices, 96, device=DEVICE),
    ]
    fm_ids = torch.tensor([0, 2], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        y = model(x, fm_ids=fm_ids)
        A = model(x, fm_ids=fm_ids, get_attention=True)

    assert y.shape == (batch_size, 96)
    assert torch.allclose(y, torch.bmm(A, x[1]).squeeze(1), atol=1e-6)


def test_cobra_raw_pooling_out_of_range_index_raises():
    batch_size, num_slices = 2, 5
    model = _build_multi_fm_raw_cobra(raw_aggregation_index=5)  # only 2 FMs will be passed
    x = [
        torch.randn(batch_size, num_slices, 64, device=DEVICE),
        torch.randn(batch_size, num_slices, 96, device=DEVICE),
    ]
    with pytest.raises(ValueError, match="raw_aggregation_index=5 is invalid"):
        model(x)


def test_select_raw_aggregation_fm_helper():
    """Unit-test the selection helper directly, including the non-list passthrough case."""
    model = _build_multi_fm_raw_cobra(raw_aggregation_index=1)
    single_tensor = torch.randn(2, 4, 64, device=DEVICE)
    assert model._select_raw_aggregation_fm(single_tensor) is single_tensor

    one_fm = [torch.randn(2, 4, 64, device=DEVICE)]
    assert model._select_raw_aggregation_fm(one_fm) is one_fm[0]

    two_fms = [torch.randn(2, 4, 64, device=DEVICE), torch.randn(2, 4, 96, device=DEVICE)]
    assert model._select_raw_aggregation_fm(two_fms) is two_fms[1]


def test_cobra_per_fm_id_inference_embed_selects_adapter_by_fm_id():
    model = _build_per_fm_cobra(mode="inference", fm_input_dims=(64, 128, 128), num_fms=3)
    batch_size, num_slices = 2, 4
    x = [
        torch.randn(batch_size, num_slices, 64, device=DEVICE),
        torch.randn(batch_size, num_slices, 128, device=DEVICE),
    ]
    fm_ids = torch.tensor([0, 2], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        out = model._embed_inference_fm_set(x, fm_ids=fm_ids)  # [K, B, D, E]
        ref0 = model.embed_fm["0"](x[0])
        ref1 = model.embed_fm["2"](x[1])
    assert out.shape == (2, batch_size, num_slices, 64)
    assert torch.allclose(out[0], ref0, atol=1e-6)
    assert torch.allclose(out[1], ref1, atol=1e-6)
