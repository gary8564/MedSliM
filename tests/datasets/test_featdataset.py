import os
from pathlib import Path
import pytest
import torch
import yaml

from med_slim.data.feat_dataset import (
    PrecomputedFeatPairDataset,
    FeatClassificationDataset,
    MultiViewFeatClassificationDataset,
    multiview_classifier_collate_fn,
    ssl_collate_fn,
    ssl_packed_collate_fn,
    linear_classifier_collate_fn,
    linear_classifier_packed_collate_fn,
    multiview_classifier_packed_collate_fn,
)
from torch.utils.data import DataLoader

# Path to pretrain config
PRETRAIN_CONFIG_PATH = Path(__file__).parent.parent.parent / "med_slim" / "configs" / "pretrain.yml"


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
    """Test the precomputed feature pair dataset matches the number of studies specified in pretrain.yml."""
    assert PRETRAIN_CONFIG_PATH.exists(), f"Pretrain config not found: {PRETRAIN_CONFIG_PATH}"
    with open(PRETRAIN_CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    datasets_cfg = cfg["feat_dataset"]["datasets"]
    models = cfg["feat_dataset"]["model_name"]
    planes = cfg["feat_dataset"]["plane"]
    split = "train"
    
    max_feature_dim = max(m["embed_dim"] for m in cfg["model"]["slice_encoder_models"])
    
    # Count unique exam_ids across all planes
    expected_counts = {}
    for ds_cfg in datasets_cfg:
        dataset_name = ds_cfg["name"]
        feat_dir = Path(ds_cfg["feat_dir"])
        
        exam_ids = set()
        plane_counts = {}
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
            
            plane_counts[plane] = len(plane_exam_ids)
            print(f"\n{dataset_name.upper()}/{plane}: {len(plane_exam_ids)} exams")
        
        expected_counts[dataset_name] = len(exam_ids)
        print(f"\n{dataset_name.upper()}: {len(exam_ids)} unique exams across all planes")
    
    total_expected = sum(expected_counts.values())
    print(f"\nTotal expected studies: {total_expected}")
    
    feat_dirs = [{"name": d["name"], "feat_dir": d["feat_dir"]} for d in datasets_cfg]
    
    ds = PrecomputedFeatPairDataset(
        feat_dirs=feat_dirs,
        slice_encoder_models=models,
        view_planes=planes,
        split=split,
        max_feature_dim=max_feature_dim,
    )
    
    loader = DataLoader(
        ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=0,
        drop_last=False,
        pin_memory=False,
        collate_fn=ssl_collate_fn,
    )
    
    print(f"Dataset length: {len(ds)}")
    print(f"DataLoader batches: {len(loader)} (batch_size={cfg['train']['batch_size']})")
    
    # Verify dataset length matches expected
    assert len(ds) == total_expected, \
        f"Dataset has {len(ds)} studies but expected {total_expected} studies"


# -------------------- Packed Collate Functions --------------------
def test_ssl_packed_collate_fn():
    """Test ssl_packed_collate_fn produces correct packed batch format."""
    assert FEAT_ROOT.exists(), f"Feature root not found: {FEAT_ROOT}"
    
    feat_dirs = [{"name": "mrnet", "feat_dir": str(FEAT_ROOT)}]
    
    ds = PrecomputedFeatPairDataset(
        feat_dirs=feat_dirs,
        slice_encoder_models=SLICE_ENCODER_MODELS[:1],  # Single encoder for simplicity
        view_planes=VIEW_PLANES,
        split=SPLIT,
        max_feature_dim=EMBED_DIMS[0],
    )
    
    if len(ds) < 4:
        pytest.skip("Need at least 4 samples for packed collate test.")
    
    batch = [ds[i] for i in range(4)]
    collated = ssl_packed_collate_fn(batch)
    
    # Check all expected keys
    expected_keys = {
        "feats1", "feats2",
        "cu_seqlens1", "cu_seqlens2",
        "max_seqlen1", "max_seqlen2",
        "seq_idx1", "seq_idx2",
        "orig_embed_dim1", "orig_embed_dim2",
        "batch_size",
    }
    assert set(collated.keys()) == expected_keys
    
    batch_size = collated["batch_size"]
    assert batch_size == 4
    
    # Verify packed tensor shapes
    total_tokens1 = collated["cu_seqlens1"][-1].item()
    total_tokens2 = collated["cu_seqlens2"][-1].item()
    
    # feats should be [total_tokens, max_feature_dim]
    assert collated["feats1"].ndim == 2
    assert collated["feats2"].ndim == 2
    assert collated["feats1"].shape[0] == total_tokens1
    assert collated["feats2"].shape[0] == total_tokens2
    
    # cu_seqlens should be [batch_size + 1]
    assert collated["cu_seqlens1"].shape == (batch_size + 1,)
    assert collated["cu_seqlens2"].shape == (batch_size + 1,)
    
    # cu_seqlens should start at 0 and be monotonically increasing
    assert collated["cu_seqlens1"][0] == 0
    assert collated["cu_seqlens2"][0] == 0
    
    # seq_idx should match total_tokens
    assert collated["seq_idx1"].shape == (total_tokens1,)
    assert collated["seq_idx2"].shape == (total_tokens2,)
    
    # Verify seq_idx values are in range [0, batch_size)
    assert collated["seq_idx1"].min() >= 0
    assert collated["seq_idx1"].max() < batch_size


def test_linear_classifier_packed_collate_fn():
    """Test linear_classifier_packed_collate_fn produces correct packed format."""
    ds = FeatClassificationDataset(
        feat_dir=str(FEAT_ROOT),
        slice_encoder_models=SLICE_ENCODER_MODELS,
        view_plane="sagittal",
        split="train",
        annotations_path=str(ANNOTATIONS_DIR / "train.csv"),
        task="binary",
        target_columns=["abnormal"],
    )
    
    if len(ds) < 4:
        pytest.skip("Need at least 4 samples for packed collate test.")
    
    batch = [ds[i] for i in range(4)]
    collated = linear_classifier_packed_collate_fn(batch)
    
    # Check keys
    expected_keys = {
        "features", "cu_seqlens", "max_seqlen", "seq_idx",
        "labels", "sample_ids", "batch_size",
    }
    assert set(collated.keys()) == expected_keys
    
    batch_size = collated["batch_size"]
    K = len(SLICE_ENCODER_MODELS)
    
    # features should be List of K tensors [total_tokens, embed_dim]
    assert len(collated["features"]) == K
    
    total_tokens = collated["cu_seqlens"][-1].item()
    for i, feat in enumerate(collated["features"]):
        assert feat.shape == (total_tokens, EMBED_DIMS[i])
    
    # cu_seqlens should be [batch_size + 1]
    assert collated["cu_seqlens"].shape == (batch_size + 1,)
    assert collated["cu_seqlens"][0] == 0
    
    # seq_idx should be [total_tokens]
    assert collated["seq_idx"].shape == (total_tokens,)
    
    # labels should be [batch_size]
    assert collated["labels"].shape == (batch_size,)


def test_multiview_classifier_packed_collate_fn():
    """Test multiview_classifier_packed_collate_fn produces correct packed format."""
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
    
    if len(ds) < 4:
        pytest.skip("Need at least 4 samples for packed collate test.")
    
    batch = [ds[i] for i in range(4)]
    collated = multiview_classifier_packed_collate_fn(batch)
    
    # Check keys
    expected_keys = {
        "features", "cu_seqlens", "max_seqlen", "seq_idx",
        "labels", "sample_ids", "batch_size",
    }
    assert set(collated.keys()) == expected_keys
    
    batch_size = collated["batch_size"]
    K = len(SLICE_ENCODER_MODELS)
    
    # Each view should have packed features
    for view in view_planes:
        assert view in collated["features"]
        assert view in collated["cu_seqlens"]
        assert view in collated["seq_idx"]
        
        features_list = collated["features"][view]
        cu_seqlens = collated["cu_seqlens"][view]
        seq_idx = collated["seq_idx"][view]
        
        assert len(features_list) == K
        
        total_tokens = cu_seqlens[-1].item()
        for i, feat in enumerate(features_list):
            assert feat.shape == (total_tokens, EMBED_DIMS[i])
        
        assert cu_seqlens.shape == (batch_size + 1,)
        assert seq_idx.shape == (total_tokens,)
    
    assert collated["labels"].shape == (batch_size,)
