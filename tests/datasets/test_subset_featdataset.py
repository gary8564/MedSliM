"""Tests for FM subset SSL mode in PrecomputedFeatPairDataset."""
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch.utils.data import DataLoader

from med_slim.data.feat_dataset import PrecomputedFeatPairDataset

SPLIT = "train"
PLANE = "sagittal"
REGIONAL_TOKENS = 4
NUM_REGIONS = 1 + REGIONAL_TOKENS

# Four FMs; duplicate feature dims (16 and 24 each appear twice) to verify that
# FM identity is tracked by global ID, not by feature dimension.
FM_DIMS = {"dinov2": 16, "rad-dino": 24, "biomedclip": 16, "ark": 24}
FM_NAMES = list(FM_DIMS.keys())
MAX_FEATURE_DIM = 24
NUM_TARGET_SLICES = 8


def _write_feat(path: Path, num_slices: int, feat_dim: int, model_name: str, tiled: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    if tiled:
        feats = torch.randn(num_slices, NUM_REGIONS, feat_dim, dtype=torch.float32)
        metadata = {
            "uid": path.stem, "plane": PLANE, "model_name": model_name,
            "slice_spacing_mm": "4.0", "token_layout": "tiled_cls",
            "regional_tokens": str(REGIONAL_TOKENS), "num_regions": str(NUM_REGIONS),
        }
    else:
        feats = torch.randn(num_slices, feat_dim, dtype=torch.float32)
        metadata = {
            "uid": path.stem, "plane": PLANE, "model_name": model_name,
            "slice_spacing_mm": "4.0",
        }
    save_file({"feats": feats}, str(path), metadata=metadata)


def _build_cache(tmp_path: Path, tiled: bool) -> str:
    feat_dir = tmp_path / ("tiled_cache" if tiled else "global_cache")
    uid_to_depth = {"0001": 5, "0002": 12, "0003": 8}
    for model_name, feat_dim in FM_DIMS.items():
        for uid, depth in uid_to_depth.items():
            _write_feat(
                feat_dir / model_name / SPLIT / PLANE / f"{uid}.safetensors",
                depth, feat_dim, model_name, tiled,
            )
    return str(feat_dir)


@pytest.fixture
def global_cache(tmp_path):
    return _build_cache(tmp_path, tiled=False)


@pytest.fixture
def tiled_cache(tmp_path):
    return _build_cache(tmp_path, tiled=True)


def _build_dataset(feat_dir, **kwargs):
    kwargs.setdefault("fm_subset_size", 2)
    kwargs.setdefault("fm_subset_min_overlap", 1)
    return PrecomputedFeatPairDataset(
        feat_dirs=[{"name": "synthetic", "feat_dir": feat_dir}],
        slice_encoder_models=FM_NAMES,
        view_planes=[PLANE],
        split=SPLIT,
        max_feature_dim=MAX_FEATURE_DIM,
        num_target_slices=NUM_TARGET_SLICES,
        ssl_fm_mode="subset",
        **kwargs,
    )


def test_subset_mode_global_item_shapes(global_cache):
    ds = _build_dataset(global_cache)
    item = ds[0]
    assert item["feats1"].shape == (2, NUM_TARGET_SLICES, MAX_FEATURE_DIM)
    assert item["feats2"].shape == (2, NUM_TARGET_SLICES, MAX_FEATURE_DIM)
    assert item["orig_embed_dim1"].shape == (2,)
    assert item["fm_ids1"].shape == (2,)
    assert item["fm_ids2"].shape == (2,)
    assert item["physical_positions"].shape == (NUM_TARGET_SLICES,)


def test_subset_mode_tiles_shapes(tiled_cache):
    ds = _build_dataset(tiled_cache)
    item = ds[0]
    assert item["feats1"].shape == (2, NUM_TARGET_SLICES, NUM_REGIONS, MAX_FEATURE_DIM)
    assert item["feats2"].shape == (2, NUM_TARGET_SLICES, NUM_REGIONS, MAX_FEATURE_DIM)


def test_subset_mode_views(global_cache):
    ds = _build_dataset(
        global_cache,
        fm_subset_size=2,
        fm_subset_min_overlap=1,
        fm_subset_max_overlap=1,
    )
    for _ in range(200):
        v1, v2 = ds._sample_fm_subsets()
        overlap = len(set(v1) & set(v2))
        assert set(v1) != set(v2)
        assert overlap == 1
        assert len(v1) == 2 and len(v2) == 2


def test_subset_mode_fm_ids(global_cache):
    ds = _build_dataset(global_cache)
    item = ds[0]
    ids1 = item["fm_ids1"]
    # orig_embed_dim follows the FM at that global id (duplicate dims handled)
    for pos, fid in enumerate(ids1.tolist()):
        assert item["orig_embed_dim1"][pos].item() == FM_DIMS[FM_NAMES[fid]]


def test_subset_mode_default_collate_batches(global_cache):
    ds = _build_dataset(global_cache)
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    assert batch["feats1"].shape == (2, 2, NUM_TARGET_SLICES, MAX_FEATURE_DIM)
    assert batch["fm_ids1"].shape == (2, 2)
    assert batch["orig_embed_dim1"].shape == (2, 2)


def test_subset_mode_size_must_be_strict_subset(global_cache):
    with pytest.raises(ValueError, match="fm_subset_size"):
        _build_dataset(global_cache, fm_subset_size=len(FM_NAMES))


def test_subset_mode_packed_not_supported(global_cache):
    with pytest.raises(NotImplementedError):
        PrecomputedFeatPairDataset(
            feat_dirs=[{"name": "synthetic", "feat_dir": global_cache}],
            slice_encoder_models=FM_NAMES,
            view_planes=[PLANE],
            split=SPLIT,
            max_feature_dim=MAX_FEATURE_DIM,
            num_target_slices=NUM_TARGET_SLICES,
            use_packed=True,
            ssl_fm_mode="subset",
            fm_subset_size=2,
        )



def test_pair_mode(global_cache):
    ds = PrecomputedFeatPairDataset(
        feat_dirs=[{"name": "synthetic", "feat_dir": global_cache}],
        slice_encoder_models=FM_NAMES,
        view_planes=[PLANE],
        split=SPLIT,
        max_feature_dim=MAX_FEATURE_DIM,
        num_target_slices=NUM_TARGET_SLICES,
    )
    item = ds[0]
    assert item["feats1"].shape == (NUM_TARGET_SLICES, MAX_FEATURE_DIM)
    assert item["orig_embed_dim1"].ndim == 0
    assert "fm_ids1" not in item
