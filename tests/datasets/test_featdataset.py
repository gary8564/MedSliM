import os
from pathlib import Path
import pytest
import torch

from med_slim.data.feat_dataset import PrecomputedFeatPairDataset, FeatClassificationDataset


FEAT_ROOT = Path("/hpcwork/rwth1833/feat_caches/MRNet")
ANNOTATIONS_DIR = Path("/hpcwork/rwth1833/datasets/preprocessed/MRNet")
SLICE_ENCODER_MODELS = ["dinov2", "rad-dino", "medsiglip", "biomedclip", "ark"]
EMBED_DIMS = [1024, 768, 1152, 512, 1376]
SPLIT = "train"
VIEW_PLANES = ["sagittal"]
MAX_FEATURE_DIM = 1376
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

    feats1, orig_dim1, feats2, orig_dim2 = ds[0]
    assert isinstance(feats1, torch.Tensor) and isinstance(feats2, torch.Tensor)
    assert feats1.ndim == 2 and feats2.ndim == 2
    assert feats1.shape == feats2.shape == (NUM_SLICES, MAX_FEATURE_DIM) # Shape: [num_slices, padded_embed_dim]
    assert isinstance(orig_dim1.item(), int) and isinstance(orig_dim2.item(), int)
    assert 1 <= orig_dim1.item() <= MAX_FEATURE_DIM # Original embedding dims are <= MAX_FEATURE_DIM
    assert 1 <= orig_dim2.item() <= MAX_FEATURE_DIM


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
    assert "feature_embeds" in item and "label" in item and "sample_id" in item
    assert len(item["feature_embeds"]) == 1  # single encoder
    assert item["feature_embeds"][0].shape == (NUM_SLICES, EMBED_DIMS[0])
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
    for i, feat in enumerate(item["feature_embeds"]):
        assert feat.shape == (NUM_SLICES, EMBED_DIMS[i])

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

