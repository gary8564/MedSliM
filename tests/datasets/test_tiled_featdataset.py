from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from med_slim.data.feat_dataset import (
    PrecomputedFeatPairDataset,
    FeatClassificationDataset,
    _validate_feature_metadata,
    linear_classifier_collate_fn,
    ssl_packed_collate_fn,
)
from med_slim.model.sequence_encoder.cobra import Cobra

DEVICE = torch.device("cpu")

SPLIT = "train"
PLANE = "sagittal"
REGIONAL_TOKENS = 4
NUM_REGIONS = 1 + REGIONAL_TOKENS  # global + 2x2
# Two FMs with different raw feature dims to exercise feature-dim padding.
FM_DIMS = {"dinov2": 16, "rad-dino": 24}
MAX_FEATURE_DIM = 24
NUM_TARGET_SLICES = 8


def _write_tiled_feat(path: Path, num_slices: int, feat_dim: int, model_name: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    feats = torch.randn(num_slices, NUM_REGIONS, feat_dim, dtype=torch.float32)
    metadata = {
        "uid": path.stem,
        "plane": PLANE,
        "model_name": model_name,
        "num_slices": "raw",
        "spatial_mode": "crop",
        "slice_spacing_mm": "4.0",
        "token_layout": "tiled_cls",
        "regional_tokens": str(REGIONAL_TOKENS),
        "tile_grid": "2x2",
        "num_regions": str(NUM_REGIONS),
        "region_order": "global,r0c0,r0c1,r1c0,r1c1",
        "include_global_cls": "true",
    }
    save_file({"feats": feats}, str(path), metadata=metadata)


@pytest.fixture
def tiled_cache(tmp_path):
    """Create a tiled cache for two FMs with several volumes of varying depth."""
    feat_dir = tmp_path / "tiled_cache"
    # Synthetic volumes with varying number of slices relative to NUM_TARGET_SLICES.
    uid_to_depth = {"0001": 5, "0002": 12, "0003": 8}
    for model_name, feat_dim in FM_DIMS.items():
        for uid, depth in uid_to_depth.items():
            _write_tiled_feat(
                feat_dir / model_name / SPLIT / PLANE / f"{uid}.safetensors",
                depth, feat_dim, model_name,
            )
    # Annotations CSV listing exactly the synthesized uids.
    annot = pd.DataFrame({"ID": list(uid_to_depth.keys()), "abnormal": [0, 1, 1]})
    annot_path = tmp_path / "annotations.csv"
    annot.to_csv(annot_path, index=False)
    return {"feat_dir": str(feat_dir), "annotations_path": str(annot_path),
            "uid_to_depth": uid_to_depth}


def test_PrecomputedFeatPairDataset(tiled_cache):
    ds = PrecomputedFeatPairDataset(
        feat_dirs=[{"name": "synthetic", "feat_dir": tiled_cache["feat_dir"]}],
        slice_encoder_models=list(FM_DIMS.keys()),
        view_planes=[PLANE],
        split=SPLIT,
        max_feature_dim=MAX_FEATURE_DIM,
        num_target_slices=NUM_TARGET_SLICES,
    )
    assert len(ds) == 3
    item = ds[0]
    feats1, feats2 = item["feats1"], item["feats2"]
    # Region axis preserved; slice axis fixed to target; feature dim padded.
    assert feats1.shape == (NUM_TARGET_SLICES, NUM_REGIONS, MAX_FEATURE_DIM)
    assert feats2.shape == (NUM_TARGET_SLICES, NUM_REGIONS, MAX_FEATURE_DIM)
    assert item["physical_positions"].shape == (NUM_TARGET_SLICES,)
    
    loader = DataLoader(ds, batch_size=2, shuffle=False)  # default collate stacks [B, num_slices, num_tiled_regions, embed_dim]
    batch = next(iter(loader))
    feats1 = batch["feats1"]
    assert feats1.shape == (2, NUM_TARGET_SLICES, NUM_REGIONS, MAX_FEATURE_DIM)

    model = Cobra(
        embed_dim=32, contrast_dim=16, input_dims=list(FM_DIMS.values()),
        num_heads=4, num_layers=1, dropout=0.0, mode="train",
        sequence_encoder="transformer", slice_pooling="abmil",
        pooling_target="post_encoder", regional_tokens=REGIONAL_TOKENS, att_dim=16,
    ).to(DEVICE).eval()
    with torch.no_grad():
        y = model(feats1, input_feature_dims=batch["orig_embed_dim1"],
                  seq_lengths=batch["seq_len"])
    assert y.shape == (2, 16)
    assert torch.isfinite(y).all()


def test_FeatClassificationDataset(tiled_cache):
    ds = FeatClassificationDataset(
        feat_dir=tiled_cache["feat_dir"],
        slice_encoder_models=["dinov2"],
        view_plane=PLANE,
        split=SPLIT,
        annotations_path=tiled_cache["annotations_path"],
        task="binary",
        target_columns=["abnormal"],
    )
    loader = DataLoader(ds, batch_size=3, shuffle=False, collate_fn=linear_classifier_collate_fn)
    batch = next(iter(loader))
    features = batch["features"]
    assert len(features) == 1
    # [B, max_seq_len, num_tiled_regions, embed_dim] with feature dim = dinov2 raw dim (not padded here)
    assert features[0].ndim == 4
    assert features[0].shape[2] == NUM_REGIONS
    assert features[0].shape[3] == FM_DIMS["dinov2"]

    model = Cobra(
        embed_dim=32, contrast_dim=16, input_dims=[FM_DIMS["dinov2"]],
        num_heads=4, num_layers=1, dropout=0.0, mode="inference",
        sequence_encoder="transformer", slice_pooling="abmil",
        pooling_target="post_encoder", regional_tokens=REGIONAL_TOKENS, att_dim=16,
    ).to(DEVICE).eval()
    with torch.no_grad():
        emb = model(features, seq_lengths=batch["seq_lengths"],
                    physical_positions=batch["physical_positions"])
    assert emb.shape == (3, 32)
    assert torch.isfinite(emb).all()


def test_PrecomputedFeatPairDataset_packed_mode(tiled_cache):
    ds = PrecomputedFeatPairDataset(
        feat_dirs=[{"name": "synthetic", "feat_dir": tiled_cache["feat_dir"]}],
        slice_encoder_models=list(FM_DIMS.keys()),
        view_planes=[PLANE],
        split=SPLIT,
        max_feature_dim=MAX_FEATURE_DIM,
        num_target_slices=NUM_TARGET_SLICES,
        use_packed=True,
    )
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=ssl_packed_collate_fn)
    batch = next(iter(loader))

    total_slices = int(batch["cu_seqlens1"][-1].item())
    assert batch["feats1"].shape == (total_slices, NUM_REGIONS, MAX_FEATURE_DIM)
    assert batch["physical_positions"].shape == (total_slices,)
    if not torch.cuda.is_available():
        pytest.skip("Packed Transformer uses FlashAttention varlen, which requires CUDA.")

    device = torch.device("cuda")

    model = Cobra(
        embed_dim=32, contrast_dim=16, input_dims=list(FM_DIMS.values()),
        num_heads=4, num_layers=1, dropout=0.0, mode="train",
        sequence_encoder="transformer", slice_pooling="abmil",
        pooling_target="post_encoder", regional_tokens=REGIONAL_TOKENS, att_dim=16,
    ).to(device).eval()
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            y = model(
                batch["feats1"].to(device),
                input_feature_dims=batch["orig_embed_dim1"].to(device),
                use_packed=True,
                cu_seqlens=batch["cu_seqlens1"].to(device),
                max_seqlen=batch["max_seqlen1"],
                seq_idx=batch["seq_idx1"].to(device),
                physical_positions=batch["physical_positions"].to(device),
            )

    assert y.shape == (2, 16)
    assert torch.isfinite(y).all()


def test_validate_feature_metadata_rejects_global_only_with_tiled_metadata():
    feats = torch.randn(8, 16)
    metadata = {"regional_tokens": "4"}
    with pytest.raises(ValueError, match="global-only"):
        _validate_feature_metadata(feats, metadata)


def test_validate_feature_metadata_accepts_consistent_tiled_metadata():
    feats = torch.randn(8, 5, 16)
    metadata = {
        "token_layout": "tiled_cls",
        "regional_tokens": "4",
        "num_regions": "5",
    }
    _validate_feature_metadata(feats, metadata)


def test_validate_feature_metadata_rejects_region_count_mismatch():
    feats = torch.randn(8, 5, 16)
    metadata = {
        "token_layout": "tiled_cls",
        "regional_tokens": "2",
        "num_regions": "5",
    }
    with pytest.raises(ValueError, match="regional_tokens metadata"):
        _validate_feature_metadata(feats, metadata)
