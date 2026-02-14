from pathlib import Path
import random

import numpy as np
import pytest
import torch
import yaml

from med_slim.data.feat_dataset import (
    PrecomputedFeatPairDataset,
    FeatClassificationDataset,
    MultiViewFeatClassificationDataset,
    multiview_classifier_collate_fn,
)
from torch.utils.data import DataLoader

_extract_study_name = PrecomputedFeatPairDataset._extract_exam_id

# Path to pretrain config
PRETRAIN_CONFIG_PATH = Path(__file__).parent.parent.parent / "med_slim" / "configs" / "pretrain.yml"


# MRNet: for classification datasets (has annotations)
FEAT_ROOT = Path("/hpcwork/rwth1833/feat_caches/MRNet/slices_raw/crop")
ANNOTATIONS_DIR = Path("/hpcwork/rwth1833/datasets/preprocessed/MRNet")
SLICE_ENCODER_MODELS = ["dinov2", "rad-dino", "dinov3"]
EMBED_DIMS = [1024, 768, 1536]
SPLIT = "train"
VIEW_PLANES = ["sagittal"]
MAX_FEATURE_DIM = 1536
NUM_SLICES = 32

# fastMRI: for PrecomputedFeatPairDataset
FEAT_ROOT_FASTMRI = Path("/hpcwork/rwth1833/feat_caches/fastMRI/slices_raw/adaptive")
FASTMRI_PLANES = ["axial", "sagittal", "coronal"]


# -------------------- PrecomputedFeatPairDataset --------------------
def _require_fastmri_feats():
    """Skip if fastMRI feature cache is not available."""
    if not FEAT_ROOT_FASTMRI.exists():
        pytest.skip(f"fastMRI feature root not found: {FEAT_ROOT_FASTMRI}")
    for model_name in SLICE_ENCODER_MODELS:
        for view_plane in FASTMRI_PLANES:
            feat_path = FEAT_ROOT_FASTMRI / model_name / SPLIT / view_plane
            if not feat_path.exists():
                pytest.skip(f"fastMRI path not found: {feat_path}")
            if len(list(feat_path.glob("*.safetensors"))) == 0:
                pytest.skip(f"No .safetensors in {feat_path}")

def test_precomputed_feat_pair_dataset_loads_and_shapes():
    """Test PrecomputedFeatPairDataset loads correctly using fastMRI (multi-series dataset)."""
    _require_fastmri_feats()
    feat_dirs = [{"name": "fastmri", "feat_dir": str(FEAT_ROOT_FASTMRI)}]

    ds = PrecomputedFeatPairDataset(
        feat_dirs=feat_dirs,
        slice_encoder_models=SLICE_ENCODER_MODELS,
        view_planes=FASTMRI_PLANES,
        split=SPLIT,
        max_feature_dim=MAX_FEATURE_DIM,
        num_target_slices=NUM_SLICES,
    )

    assert len(ds) > 0

    item = ds[0]
    feats1 = item["feats1"]
    feats2 = item["feats2"]
    orig_dim1 = item["orig_embed_dim1"]
    orig_dim2 = item["orig_embed_dim2"]
    seq_len = item["seq_len"]
    
    assert isinstance(feats1, torch.Tensor) and isinstance(feats2, torch.Tensor)
    assert feats1.ndim == 2 and feats2.ndim == 2
    # Slice sampling ensures fixed output shape: [num_target_slices, padded_embed_dim]
    assert feats1.shape == (NUM_SLICES, MAX_FEATURE_DIM)
    assert feats2.shape == (NUM_SLICES, MAX_FEATURE_DIM)
    assert isinstance(orig_dim1.item(), int) and isinstance(orig_dim2.item(), int)
    assert 1 <= orig_dim1.item() <= MAX_FEATURE_DIM  # Original embedding dims are <= MAX_FEATURE_DIM
    assert 1 <= orig_dim2.item() <= MAX_FEATURE_DIM
    assert isinstance(seq_len.item(), int)
    assert 1 <= seq_len.item() <= NUM_SLICES


def test_precomputed_feat_pair_dataset_same_patient_per_pair():
    """
    Verify that after randomly selecting view plane and MRI sequence, both feats1 and feats2
    in a pair come from the same patient/study (same UID). Tests refactored logic for
    multi-series datasets like fastMRI where each patient can have multiple sequences per plane.
    """
    _require_fastmri_feats()

    feat_dirs = [{"name": "fastmri", "feat_dir": str(FEAT_ROOT_FASTMRI)}]
    ds = PrecomputedFeatPairDataset(
        feat_dirs=feat_dirs,
        slice_encoder_models=SLICE_ENCODER_MODELS,
        view_planes=FASTMRI_PLANES,
        split=SPLIT,
        max_feature_dim=MAX_FEATURE_DIM,
        num_target_slices=NUM_SLICES,
    )

    # Check we have multi-series studies (fastMRI has multiple sequences per patient)
    studies_with_multi_series = sum(
        1 for planes in ds.feat_path_dict.values()
        if any(len(uid_list) > 1 for uid_list in planes.values())
    )
    assert studies_with_multi_series > 0, (
        "fastMRI should have studies with multiple series per plane; "
        "cannot properly test same-patient logic without multi-series data."
    )

    # Sample many pairs with fixed seed; if feats1/feats2 used different UIDs, slice count
    # could differ and __getitem__ would assert. No failures => same UID used for both.
    random.seed(42)
    np.random.seed(42)
    n_samples = min(200, len(ds))
    indices = [random.randint(0, len(ds) - 1) for _ in range(n_samples)]
    for idx in indices:
        item = ds[idx]
        assert item["feats1"].shape[0] == item["feats2"].shape[0], (
            f"Pair should have same slice count (same patient/sequence); "
            f"got {item['feats1'].shape[0]} vs {item['feats2'].shape[0]}"
        )


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


def test_multiview_classifier_collate_fn():
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
    collated = multiview_classifier_collate_fn(batch)

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


# -------------------- Pretrain Config Validation --------------------
def test_dataloader_matches_number_of_studies():
    """
    Test the precomputed feature pair dataset matches the number of patient exams specified in pretrain.yml.
    """
    assert PRETRAIN_CONFIG_PATH.exists(), f"Pretrain config not found: {PRETRAIN_CONFIG_PATH}"
    with open(PRETRAIN_CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    datasets_cfg = cfg["feat_dataset"]["datasets"]
    models = cfg["feat_dataset"]["model_name"]
    planes = cfg["feat_dataset"]["plane"]
    split = "train"
    
    max_feature_dim = max(m["embed_dim"] for m in cfg["model"]["slice_encoder_models"])
    
    # Count unique patient exam_ids across all planes
    expected_counts = {}
    for ds_cfg in datasets_cfg:
        dataset_name = ds_cfg["name"]
        feat_dir = Path(ds_cfg["feat_dir"])
        
        study_names = set()
        exam_ids = set()
        for plane in planes:
            ref_path = feat_dir / models[0] / split / plane
            if not ref_path.exists():
                print(f"\n{plane}: path not found ({ref_path})")
                continue
            
            plane_exam_ids = set()
            for f in ref_path.glob("*.safetensors"):
                exam_id = f.stem
                plane_exam_ids.add(exam_id)
                exam_ids.add(exam_id)
                study_name = _extract_study_name(exam_id)
                study_names.add(f"{dataset_name}_{study_name}")
            
            print(f"\n{dataset_name.upper()}/{plane}: {len(plane_exam_ids)} series")
        
        expected_counts[dataset_name] = len(study_names)
        print(f"\n{dataset_name.upper()}: {len(exam_ids)} total series with {len(study_names)} unique patient exams")
    
    total_expected = sum(expected_counts.values())
    print(f"\nTotal expected patient exams: {total_expected}")
    
    feat_dirs = [{"name": d["name"], "feat_dir": d["feat_dir"]} for d in datasets_cfg]
    
    num_target_slices = cfg["feat_dataset"].get("num_target_slices", 32)
    
    ds = PrecomputedFeatPairDataset(
        feat_dirs=feat_dirs,
        slice_encoder_models=models,
        view_planes=planes,
        split=split,
        max_feature_dim=max_feature_dim,
        num_target_slices=num_target_slices,
    )
    
    loader = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=0,
        drop_last=False,
        pin_memory=False,
    )
    
    print(f"Dataset length (patient studies): {len(ds)}")
    print(f"DataLoader batches: {len(loader)} (batch_size={cfg['train']['batch_size']})")
    
    # Verify dataset length matches expected number of patient studies
    assert len(ds) == total_expected, \
        f"Dataset has {len(ds)} patient studies but expected {total_expected}"


