import os
from pathlib import Path
import pytest
import torch

from med_slim.data.feat_dataset import (
    PrecomputedFeatPairDataset,
    FeatClassificationDataset,
    MultiViewFeatClassificationDataset,
    multiview_collate_fn,
)


FEAT_ROOT = Path("/hpcwork/rwth1833/feat_caches/MRNet/slices_raw")
ANNOTATIONS_DIR = Path("/hpcwork/rwth1833/datasets/preprocessed/MRNet")
SLICE_ENCODER_MODELS = ["dinov2", "rad-dino", "dinov3"]
EMBED_DIMS = [1024, 768, 1536]
SPLIT = "train"
VIEW_PLANES = ["sagittal"]
MAX_FEATURE_DIM = 1536
NUM_SLICES = 32


# -------------------- PrecomputedFeatPairDataset --------------------
def test_precomputed_feat_pair_dataset_loads_and_shapes():
    assert FEAT_ROOT.exists(), f"Feature root not found: {FEAT_ROOT}"
    for model_name in SLICE_ENCODER_MODELS:
        for view_plane in VIEW_PLANES:
            feat_path = FEAT_ROOT / model_name / SPLIT / view_plane
            assert feat_path.exists(), f"Expected path not found: {feat_path}"
            feat_files = list(feat_path.glob("*.safetensors"))
            assert len(feat_files) > 0, f"No .safetensors files found under {feat_path}"
            
    feat_dirs = [{"name": "mrnet", "feat_dir": str(FEAT_ROOT)}]

    ds = PrecomputedFeatPairDataset(
        feat_dirs=feat_dirs,
        slice_encoder_models=SLICE_ENCODER_MODELS,
        view_planes=VIEW_PLANES,
        split=SPLIT,
        max_feature_dim=MAX_FEATURE_DIM,
    )

    assert len(ds) > 0

    feats1, orig_dim1, seq_len1, feats2, orig_dim2, seq_len2 = ds[0]
    assert isinstance(feats1, torch.Tensor) and isinstance(feats2, torch.Tensor)
    assert feats1.ndim == 2 and feats2.ndim == 2
    assert feats1.shape == feats2.shape == (NUM_SLICES, MAX_FEATURE_DIM) # Shape: [num_slices, padded_embed_dim]
    assert isinstance(orig_dim1.item(), int) and isinstance(orig_dim2.item(), int)
    assert 1 <= orig_dim1.item() <= MAX_FEATURE_DIM # Original embedding dims are <= MAX_FEATURE_DIM
    assert 1 <= orig_dim2.item() <= MAX_FEATURE_DIM
    assert isinstance(seq_len1.item(), int) and isinstance(seq_len2.item(), int)
    assert seq_len1.item() == seq_len2.item()  # Both views should have same sequence length


# -------------------- FeatClassificationDataset --------------------
def test_feat_classification_dataset_single_encoder():
    ds = FeatClassificationDataset(
        feat_dir=str(FEAT_ROOT),
        slice_encoder_models=["dinov2"],
        view_plane="sagittal",
        split="train",
        annotations_path=str(ANNOTATIONS_DIR / "train.csv"),
        task="binary",
        target_columns=["abnormal"],
    )
    assert len(ds) > 0
    
    item = ds[0]
    assert "feature_embeds" in item and "label" in item and "sample_id" in item and "seq_length" in item
    assert len(item["feature_embeds"]) == 1  # single encoder
    # Dataset returns raw variable-length features with shape [seq_length, embed_dim]
    seq_len = item["seq_length"]
    assert item["feature_embeds"][0].shape == (seq_len, EMBED_DIMS[0])
    assert item["label"].dtype == torch.long  # binary classification task

def test_feat_classification_dataset_multi_encoder():
    ds = FeatClassificationDataset(
        feat_dir=str(FEAT_ROOT),
        slice_encoder_models=SLICE_ENCODER_MODELS,
        view_plane="sagittal",
        split="train",
        annotations_path=str(ANNOTATIONS_DIR / "train.csv"),
        task="binary",
        target_columns=["abnormal"]
    )
    assert len(ds) > 0
    
    item = ds[0]
    assert len(item["feature_embeds"]) == len(SLICE_ENCODER_MODELS)
    seq_len = item["seq_length"]
    for i, feat in enumerate(item["feature_embeds"]):
        # All encoders should have same seq_len at the same exam level
        assert feat.shape == (seq_len, EMBED_DIMS[i])

def test_feat_classification_dataset_multilabel():
    ds = FeatClassificationDataset(
        feat_dir=str(FEAT_ROOT),
        slice_encoder_models=["dinov2"],
        view_plane="sagittal",
        split="train",
        annotations_path=str(ANNOTATIONS_DIR / "train.csv"),
        task="multilabel",
        target_columns=["abnormal", "acl", "meniscus"],
    )
    item = ds[0]
    assert item["label"].shape == (3,)  # 3 targets
    assert item["label"].dtype == torch.float32  # multilabel uses float

def test_feat_classification_dataset_invalid_task():
    with pytest.raises(ValueError, match="task"):
        FeatClassificationDataset(
            feat_dir=str(FEAT_ROOT),
            slice_encoder_models=["dinov2"],
            view_plane="sagittal",
            split="train",
            annotations_path=str(ANNOTATIONS_DIR / "train.csv"),
            task="invalid_task",
            target_columns=["abnormal"],
        )


# -------------------- MultiViewFeatClassificationDataset --------------------
def _get_available_views(min_views: int = 2):
    first_encoder_dir = FEAT_ROOT / SLICE_ENCODER_MODELS[0] / SPLIT
    if not first_encoder_dir.exists():
        return []
    available = sorted(
        [p.name for p in first_encoder_dir.iterdir() if p.is_dir()]
    )
    return available if len(available) >= min_views else []


def test_multiview_feat_classification_dataset():
    view_planes = _get_available_views(min_views=2)
    if not view_planes:
        pytest.skip("Not enough view planes available on disk for multi-view test.")

    ds = MultiViewFeatClassificationDataset(
        feat_dir=str(FEAT_ROOT),
        slice_encoder_models=SLICE_ENCODER_MODELS,
        view_planes=view_planes,
        split="train",
        annotations_path=str(ANNOTATIONS_DIR / "train.csv"),
        task="binary",
        target_columns=["abnormal"],
    )

    assert len(ds) > 0
    item = ds[0]

    assert set(view_planes).issubset(item["feature_embeds"].keys())
    assert set(view_planes).issubset(item["seq_length"].keys())
    assert item["label"].dtype == torch.long

    for view in view_planes:
        seq_len = item["seq_length"][view]
        feats_for_view = item["feature_embeds"][view]
        assert len(feats_for_view) == len(SLICE_ENCODER_MODELS)
        # All encoders share the same sequence length for this view
        for i, feat in enumerate(feats_for_view):
            assert feat.shape == (seq_len, EMBED_DIMS[i])


def test_multiview_collate_fn():
    view_planes = _get_available_views(min_views=2)
    if not view_planes:
        pytest.skip("Not enough view planes available on disk for multi-view test.")

    ds = MultiViewFeatClassificationDataset(
        feat_dir=str(FEAT_ROOT),
        slice_encoder_models=SLICE_ENCODER_MODELS,
        view_planes=view_planes,
        split="train",
        annotations_path=str(ANNOTATIONS_DIR / "train.csv"),
        task="binary",
        target_columns=["abnormal"],
    )

    if len(ds) < 2:
        pytest.skip("Need at least two samples for collate function test.")

    batch = [ds[0], ds[1]]
    collated = multiview_collate_fn(batch)

    assert set(collated.keys()) == {"features", "seq_lengths", "labels", "sample_ids"}

    for view in view_planes:
        features_list = collated["features"][view]
        seq_lengths = collated["seq_lengths"][view]
        assert len(features_list) == len(SLICE_ENCODER_MODELS)
        assert seq_lengths.shape[0] == len(batch)

        max_seq = seq_lengths.max().item()
        for i, feat in enumerate(features_list):
            # Each tensor should be [B, max_seq_len, embed_dim]
            assert feat.shape == (len(batch), max_seq, EMBED_DIMS[i])

