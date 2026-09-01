"""AnyMC3D baseline: task-query pooling and frozen-FM classifier wiring."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from safetensors.torch import save_file
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

from baselines.anymc3d.classifier import (
    AnyMC3DClassifier,
    _derive_fold_seed,
    _fold_splits,
    _model_inputs,
    resolve_model_name,
)
from baselines.anymc3d.pooling import TaskQueryPooling, lengths_to_mask, slice_embeddings
from med_slim.data.feat_dataset import FeatClassificationDataset, linear_classifier_collate_fn

EMBED_DIM = 8
PLANE = "sagittal"
MODEL_NAME = "mri-core"


def test_query_pooling_matches_paper_formula():
    torch.manual_seed(0)
    pool = TaskQueryPooling(embed_dim=EMBED_DIM, query_init_std=0.02)
    hidden = torch.randn(2, 5, EMBED_DIM)
    mask = torch.tensor(
        [[True, True, True, False, False], [True, True, True, True, False]]
    )

    pooled, attn = pool(hidden, mask=mask, return_attention=True)

    scale = EMBED_DIM ** 0.5
    scores = torch.matmul(hidden, pool.query.detach()) / scale
    scores = scores.masked_fill(~mask, float("-inf"))
    expected_attn = torch.softmax(scores, dim=-1)
    expected = torch.bmm(expected_attn.unsqueeze(1), hidden).squeeze(1)

    torch.testing.assert_close(attn, expected_attn)
    torch.testing.assert_close(pooled, expected)
    assert torch.allclose(attn.sum(dim=-1), torch.ones(2), atol=1e-5)
    assert torch.all(attn[:, 3:][0] == 0)
    assert attn[1, 4] == 0


def test_query_pooling_attends_to_matching_slice():
    pool = TaskQueryPooling(embed_dim=EMBED_DIM, query_init_std=0.02)
    with torch.no_grad():
        pool.query.copy_(torch.ones(EMBED_DIM))
    hidden = torch.zeros(1, 4, EMBED_DIM)
    hidden[0, 2] = pool.query.detach() * 8
    _, attn = pool(hidden, return_attention=True)
    assert int(attn.argmax(dim=-1).item()) == 2
    assert attn[0, 2] > 0.9


def test_slice_embeddings_rejects_multi_fm_and_tiled_caches():
    with pytest.raises(ValueError, match="one frozen 2D FM"):
        slice_embeddings([torch.randn(2, 3, 4), torch.randn(2, 3, 4)])

    tiled = torch.randn(2, 5, 4, 8)
    with pytest.raises(ValueError, match="global-only"):
        slice_embeddings(tiled)


def test_resolve_model_name_is_single_fm():
    assert resolve_model_name("mri-core") == "mri-core"
    assert resolve_model_name(["mri-core"]) == "mri-core"
    with pytest.raises(ValueError, match="one frozen 2D FM"):
        resolve_model_name(["mri-core", "dinov2"])


def test_classifier_shapes_and_query_gradients():
    model = AnyMC3DClassifier(embed_dim=EMBED_DIM, num_classes=3)
    features = [torch.randn(2, 6, EMBED_DIM)]
    seq_lengths = torch.tensor([6, 4])
    out = model(features, seq_lengths, return_attention=True)
    assert out["logits"].shape == (2, 3)
    assert out["pooled"].shape == (2, EMBED_DIM)
    assert out["attn_weights"].shape == (2, 6)
    out["logits"].sum().backward()
    assert model.pool.query.grad is not None
    assert model.pool.query.grad.abs().sum() > 0


def test_lengths_to_mask():
    mask = lengths_to_mask(torch.tensor([3, 1]), max_seq_len=4)
    expected = torch.tensor(
        [[True, True, True, False], [True, False, False, False]]
    )
    assert torch.equal(mask, expected)


def _write_feat(path: Path, num_slices: int, feat_dim: int, model_name: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    feats = torch.randn(num_slices, feat_dim, dtype=torch.float32)
    metadata = {
        "uid": path.stem,
        "plane": PLANE,
        "model_name": model_name,
        "slice_spacing_mm": "4.0",
    }
    save_file({"feats": feats}, str(path), metadata=metadata)


def test_classifier_reads_feat_classification_cache(tmp_path):
    feat_dir = tmp_path / "cache"
    depths = {"0001": 5, "0002": 8, "0003": 4}
    for uid, depth in depths.items():
        _write_feat(
            feat_dir / MODEL_NAME / "train" / PLANE / f"{uid}.safetensors",
            depth, EMBED_DIM, MODEL_NAME,
        )
    annot = pd.DataFrame({"ID": list(depths), "acl": [0, 1, 2]})
    annot_path = tmp_path / "train_multiclass.csv"
    annot.to_csv(annot_path, index=False)

    dataset = FeatClassificationDataset(
        feat_dir=str(feat_dir),
        slice_encoder_models=[MODEL_NAME],
        view_plane=PLANE,
        split="train",
        annotations_path=str(annot_path),
        task="multiclass",
        target_columns=["acl"],
    )
    loader = DataLoader(dataset, batch_size=3, collate_fn=linear_classifier_collate_fn)
    batch = next(iter(loader))
    model = AnyMC3DClassifier(embed_dim=EMBED_DIM, num_classes=3)
    out = model(**_model_inputs(batch))
    assert out["logits"].shape == (3, 3)
    assert batch["features"][0].shape[0] == 3
    assert batch["features"][0].shape[-1] == EMBED_DIM


def test_fold_seed_matches_linear_probing_repeat_seed():
    expected = int(np.random.SeedSequence(42).generate_state(1)[0])
    assert _derive_fold_seed(42) == expected


def test_fold_splits_match_sklearn_stratified_kfold():
    labels = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    seed = _derive_fold_seed(42)
    ours = _fold_splits(labels, n_folds=3, task="multiclass", seed=seed)
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    expected = list(kfold.split(np.arange(len(labels)), labels))
    assert len(ours) == 3
    for (train_a, val_a), (train_b, val_b) in zip(ours, expected):
        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(val_a, val_b)
