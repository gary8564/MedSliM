"""Curia baseline: token dispatch, official cache, and classifier head."""

import math

import numpy as np
import pytest
import torch
import torchio as tio
from sklearn.model_selection import StratifiedKFold

from baselines.curia.feature_cache import (
    CENTER_PATCH_TOKENS,
    CLS_PER_SLICE,
    PATCH_MEAN_PER_SLICE,
    CuriaTokenCacheDataset,
    assert_cache_manifest_compatible,
    cache_manifest,
    curia_cache_dir,
    curia_token_collate_fn,
    encode_curia_volume,
    encode_slice_batch,
    preprocess_curia_volume,
    read_curia_cache_entry,
    write_cache_manifest,
    write_curia_cache_entry,
)
from baselines.curia.classifier import (
    CuriaClassifier,
    LabeledCuriaTokenDataset,
    _derive_fold_seed,
    _fold_splits,
    _model_inputs,
    _train_val_indices,
)
from baselines.curia.utils import (
    assemble_volume_tokens,
    describe_token_layout,
    extract_curia_volume_tokens,
    resolve_patch_pooling,
    select_slice_indices,
    slice_positional_embeddings,
    validate_token_source,
)
from med_slim.model.attention_pooling.cross_attention import (
    InterSliceAggregator,
    validate_attention_blocks,
)
from med_slim.utils.preprocessing.augmentation import PerSliceZScore

EMBED_DIM = 8
PATCHES = 4


# Reference implementations copied from raidium-med/curia modeling_dinov2.py
def _official_slice_range(depth: int, num_slices: int | None) -> list[int]:
    if num_slices is not None:
        middle = depth // 2
        start = middle - num_slices // 2
        slice_range = range(start, start + num_slices)
    else:
        slice_range = range(depth)
    return [i for i in slice_range if 0 <= i < depth]


def _official_slice_pe(num_slices: int, dim: int) -> torch.Tensor:
    position = torch.arange(num_slices).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pos_embed = torch.zeros(num_slices, dim)
    pos_embed[:, 0::2] = torch.sin(position * div_term)
    pos_embed[:, 1::2] = torch.cos(position * div_term)
    return pos_embed


def _official_curia_preprocess(
    volume_hwd: torch.Tensor, crop_size: int = 16, eps: float = 1e-6
) -> torch.Tensor:
    """Published 3D processor branch: bicubic resize, whole-volume z-score."""
    resized = [
        torch.nn.functional.interpolate(
            volume_hwd[..., i].float()[None, None],
            size=(crop_size, crop_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )[0]
        for i in range(volume_hwd.shape[-1])
    ]
    stacked = torch.stack(resized, dim=0)
    mean, std = float(stacked.mean()), float(stacked.std())
    if std < eps:
        return stacked - mean
    return (stacked - mean) / std


def _tokens(depth: int = 5, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    cls = torch.randn(1, depth, EMBED_DIM, generator=generator)
    patch = torch.randn(1, depth, PATCHES, EMBED_DIM, generator=generator)
    return cls, patch


# token_source validation
def test_token_source_canonical_order_ignores_list_order():
    assert validate_token_source(["cls", "patch"]) == ("cls", "patch")
    assert validate_token_source(["patch", "cls"]) == ("cls", "patch")
    assert validate_token_source(["patch"]) == ("patch",)
    assert validate_token_source(("cls",)) == ("cls",)


@pytest.mark.parametrize("bad", [["token"], ["class"], ["cls_patch"], [""], ["CLS"], ["patch", None]])
def test_token_source_rejects_unknown_entries(bad):
    with pytest.raises(ValueError, match="token_source"):
        validate_token_source(bad)


def test_token_source_rejects_duplicates_empty_and_non_list():
    with pytest.raises(ValueError, match="repeat"):
        validate_token_source(["cls", "cls"])
    with pytest.raises(ValueError, match="empty"):
        validate_token_source([])
    with pytest.raises(ValueError, match="must be a list"):
        validate_token_source("cls")
    with pytest.raises(ValueError, match="must be a list"):
        validate_token_source({"cls": True})


def test_resolve_patch_pooling_follows_official_dispatch():
    assert resolve_patch_pooling(["patch"]) == "raw"
    assert resolve_patch_pooling(["patch"], use_avgpool_per_slice=True) == "per_slice"
    assert resolve_patch_pooling(["patch"], use_avgpool_on_the_volume=True) == "volume"
    # Upstream checks per-slice first, so it wins when both flags are set.
    assert resolve_patch_pooling(
        ["patch"], use_avgpool_per_slice=True, use_avgpool_on_the_volume=True
    ) == "per_slice"
    # Patch pooling flags are irrelevant without patch tokens.
    assert resolve_patch_pooling(["cls"], use_avgpool_per_slice=True) == "cls_only"


def test_describe_token_layout_reports_resolved_recipe():
    layout = describe_token_layout(["patch", "cls"], use_avgpool_per_slice=True, num_slices=None)
    assert layout == {
        "token_source": ["cls", "patch"],
        "pooling_mode": "per_slice",
        "num_slices": "all",
    }


# Parity with upstream helpers
@pytest.mark.parametrize("depth", [3, 4, 10, 29, 30])
@pytest.mark.parametrize("num_slices", [None, 1, 3, 5])
def test_select_slice_indices_matches_official_get_slice_range(depth, num_slices):
    assert select_slice_indices(depth, num_slices) == _official_slice_range(depth, num_slices)


def test_center_three_window_is_the_middle_of_the_stack():
    assert select_slice_indices(30, 3) == [14, 15, 16]
    assert select_slice_indices(31, 3) == [14, 15, 16]
    assert select_slice_indices(2, 3) == [0, 1]  # window clipped to the volume


def test_slice_positional_embeddings_match_official():
    ours = slice_positional_embeddings(6, EMBED_DIM)
    torch.testing.assert_close(ours, _official_slice_pe(6, EMBED_DIM))


# Token dispatch
def test_cls_only_keeps_one_token_per_selected_slice():
    cls, patch = _tokens(depth=9)
    tokens, mask = extract_curia_volume_tokens(
        cls_per_slice=cls, token_source=["cls"], num_slices=3
    )
    assert tokens.shape == (1, 3, EMBED_DIM)
    assert mask.all()
    expected = cls[0, [3, 4, 5]] + _official_slice_pe(3, EMBED_DIM)
    torch.testing.assert_close(tokens[0], expected)


def test_raw_patches_flatten_every_selected_slice():
    cls, patch = _tokens(depth=9)
    tokens, _ = extract_curia_volume_tokens(
        patch_per_slice=patch, token_source=["patch"], num_slices=3
    )
    assert tokens.shape == (1, 3 * PATCHES, EMBED_DIM)
    pe = _official_slice_pe(3, EMBED_DIM)
    expected = (patch[0, [3, 4, 5]] + pe.unsqueeze(1)).reshape(3 * PATCHES, EMBED_DIM)
    torch.testing.assert_close(tokens[0], expected)


def test_per_slice_mean_gives_one_patch_token_per_slice():
    cls, patch = _tokens(depth=6)
    tokens, _ = extract_curia_volume_tokens(
        patch_per_slice=patch, token_source=["patch"], use_avgpool_per_slice=True
    )
    assert tokens.shape == (1, 6, EMBED_DIM)
    expected = patch[0].mean(dim=1) + _official_slice_pe(6, EMBED_DIM)
    torch.testing.assert_close(tokens[0], expected)


def test_volume_mean_gives_a_single_token_without_positional_embedding():
    cls, patch = _tokens(depth=6)
    tokens, _ = extract_curia_volume_tokens(
        patch_per_slice=patch, token_source=["patch"], use_avgpool_on_the_volume=True
    )
    assert tokens.shape == (1, 1, EMBED_DIM)
    torch.testing.assert_close(tokens[0, 0], patch[0].reshape(-1, EMBED_DIM).mean(dim=0))


def test_volume_mean_equals_mean_of_per_slice_means():
    """Whole-volume mean equals the mean of per-slice means (cache recovery)."""
    _, patch = _tokens(depth=7)
    per_slice_mean = patch.mean(dim=2, keepdim=True)
    from_raw, _ = extract_curia_volume_tokens(
        patch_per_slice=patch, token_source=["patch"], use_avgpool_on_the_volume=True
    )
    from_means, _ = extract_curia_volume_tokens(
        patch_per_slice=per_slice_mean, token_source=["patch"], use_avgpool_on_the_volume=True
    )
    torch.testing.assert_close(from_raw, from_means)


# CLS + patch mixing
def test_mixed_raw_prepends_cls_per_slice_in_official_order():
    cls, patch = _tokens(depth=9)
    tokens, _ = extract_curia_volume_tokens(
        cls_per_slice=cls, patch_per_slice=patch, token_source=["cls", "patch"], num_slices=3
    )
    assert tokens.shape == (1, 3 * (PATCHES + 1), EMBED_DIM)
    pe = _official_slice_pe(3, EMBED_DIM)
    for position, slice_idx in enumerate([3, 4, 5]):
        start = position * (PATCHES + 1)
        torch.testing.assert_close(tokens[0, start], cls[0, slice_idx] + pe[position])
        torch.testing.assert_close(
            tokens[0, start + 1 : start + 1 + PATCHES], patch[0, slice_idx] + pe[position]
        )


def test_mixed_per_slice_interleaves_cls_then_patch_mean():
    cls, patch = _tokens(depth=4)
    tokens, _ = extract_curia_volume_tokens(
        cls_per_slice=cls,
        patch_per_slice=patch,
        token_source=["cls", "patch"],
        use_avgpool_per_slice=True,
    )
    assert tokens.shape == (1, 2 * 4, EMBED_DIM)
    pe = _official_slice_pe(4, EMBED_DIM)
    torch.testing.assert_close(tokens[0, 0::2], cls[0] + pe)
    torch.testing.assert_close(tokens[0, 1::2], patch[0].mean(dim=1) + pe)


def test_mixed_volume_pools_cls_over_the_same_slices_as_patches():
    cls, patch = _tokens(depth=9)
    tokens, _ = extract_curia_volume_tokens(
        cls_per_slice=cls,
        patch_per_slice=patch,
        token_source=["cls", "patch"],
        use_avgpool_on_the_volume=True,
        num_slices=3,
    )
    assert tokens.shape == (1, 2, EMBED_DIM)
    torch.testing.assert_close(tokens[0, 0], cls[0, [3, 4, 5]].mean(dim=0))
    torch.testing.assert_close(
        tokens[0, 1], patch[0, [3, 4, 5]].reshape(-1, EMBED_DIM).mean(dim=0)
    )


@pytest.mark.parametrize(
    "flags",
    [
        {},
        {"use_avgpool_per_slice": True},
        {"use_avgpool_on_the_volume": True},
    ],
)
def test_mixing_keeps_the_embedding_dim_and_ignores_list_order(flags):
    cls, patch = _tokens(depth=9)
    common = dict(
        cls_per_slice=cls, patch_per_slice=patch, num_slices=3, **flags
    )
    forward, _ = extract_curia_volume_tokens(token_source=["cls", "patch"], **common)
    reversed_order, _ = extract_curia_volume_tokens(token_source=["patch", "cls"], **common)
    torch.testing.assert_close(forward, reversed_order)
    assert forward.shape[-1] == EMBED_DIM


def test_positional_embeddings_can_be_disabled():
    cls, patch = _tokens(depth=4)
    tokens, _ = extract_curia_volume_tokens(
        cls_per_slice=cls,
        token_source=["cls"],
        add_slice_positional_embedding=False,
    )
    torch.testing.assert_close(tokens[0], cls[0])


def test_assemble_volume_tokens_rejects_unknown_mode():
    cls, patch = _tokens(depth=2)
    with pytest.raises(ValueError, match="pooling mode"):
        assemble_volume_tokens(cls[0], patch[0], mode="mean")


def test_missing_tensors_and_bad_shapes_raise():
    cls, patch = _tokens(depth=3)
    with pytest.raises(ValueError, match="cls_per_slice is None"):
        extract_curia_volume_tokens(patch_per_slice=patch, token_source=["cls"])
    with pytest.raises(ValueError, match="patch_per_slice is None"):
        extract_curia_volume_tokens(cls_per_slice=cls, token_source=["patch"])
    with pytest.raises(ValueError, match=r"must be \[B, D, P, E\]"):
        extract_curia_volume_tokens(patch_per_slice=cls, token_source=["patch"])
    with pytest.raises(ValueError, match="embedding dim"):
        extract_curia_volume_tokens(
            cls_per_slice=cls,
            patch_per_slice=torch.randn(1, 3, PATCHES, EMBED_DIM + 1),
            token_source=["cls", "patch"],
        )


def test_slice_mask_limits_selection_and_pads_the_batch():
    cls = torch.randn(2, 6, EMBED_DIM)
    slice_mask = torch.tensor([[True] * 6, [True] * 4 + [False] * 2])
    tokens, mask = extract_curia_volume_tokens(
        cls_per_slice=cls, slice_mask=slice_mask, token_source=["cls"]
    )
    assert tokens.shape == (2, 6, EMBED_DIM)
    assert mask[0].all()
    assert mask[1].tolist() == [True] * 4 + [False] * 2
    assert torch.count_nonzero(tokens[1, 4:]) == 0
    # The short volume centres its window on its own valid depth.
    torch.testing.assert_close(tokens[1, :4], cls[1, :4] + _official_slice_pe(4, EMBED_DIM))


# Official backbone encoding
def test_encode_slice_batch_splits_cls_and_patch_tokens():
    class _StubBackbone:
        def __init__(self, tokens):
            self.tokens = tokens

        def __call__(self, pixel_values, return_dict=True):
            num_slices = pixel_values.shape[0]
            return type("Output", (), {"last_hidden_state": self.tokens[:num_slices]})

    hidden = torch.arange((PATCHES + 1) * EMBED_DIM * 3, dtype=torch.float32).reshape(
        3, PATCHES + 1, EMBED_DIM
    )
    cls_tokens, patch_tokens = encode_slice_batch(
        _StubBackbone(hidden), torch.zeros(3, 1, 8, 8)
    )
    assert cls_tokens.shape == (3, EMBED_DIM)
    assert patch_tokens.shape == (3, PATCHES, EMBED_DIM)
    torch.testing.assert_close(cls_tokens, hidden[:, 0])
    torch.testing.assert_close(patch_tokens, hidden[:, 1:])


def test_encode_slice_batch_rejects_non_grayscale_slices():
    class _StubBackbone:
        def __call__(self, pixel_values, return_dict=True):
            raise AssertionError("backbone should not be called")

    with pytest.raises(ValueError, match=r"\[N, 1, H, W\]"):
        encode_slice_batch(_StubBackbone(), torch.zeros(3, 8, 8))
    with pytest.raises(ValueError, match="grayscale"):
        encode_slice_batch(_StubBackbone(), torch.zeros(3, 3, 8, 8))


def test_encode_curia_volume_chunks_by_slice_batch_size():
    class _CountingBackbone(torch.nn.Module):
        def __init__(self, hidden):
            super().__init__()
            self.hidden = hidden
            self.batch_sizes = []
            self.dummy = torch.nn.Parameter(torch.zeros(1))

        def forward(self, pixel_values, return_dict=True):
            self.batch_sizes.append(int(pixel_values.shape[0]))
            return type("Output", (), {"last_hidden_state": self.hidden[: pixel_values.shape[0]]})

    hidden = torch.zeros(5, PATCHES + 1, EMBED_DIM)
    backbone = _CountingBackbone(hidden)
    cls_tokens, patch_tokens = encode_curia_volume(
        backbone, torch.zeros(5, 1, 8, 8), slice_batch_size=2
    )
    assert backbone.batch_sizes == [2, 2, 1]
    assert cls_tokens.shape == (5, EMBED_DIM)
    assert patch_tokens.shape == (5, PATCHES, EMBED_DIM)
    assert cls_tokens.device.type == "cpu"


# Preprocessing: official whole-volume z-score vs MedSliM per-slice z-score
def _volume_with_slice_offsets(depth: int = 3, size: int = 8) -> torch.Tensor:
    generator = torch.Generator().manual_seed(3)
    slices = [
        torch.randn(size, size, generator=generator) * (5.0 * (i + 1)) + 100.0 * (i + 1)
        for i in range(depth)
    ]
    return torch.stack(slices, dim=-1)  # (H, W, D)


def test_official_preprocessing_normalizes_the_whole_volume_not_each_slice():
    volume = _volume_with_slice_offsets()
    processed = _official_curia_preprocess(volume, crop_size=16)

    assert processed.mean().abs() < 1e-4
    assert abs(float(processed.std()) - 1.0) < 1e-4
    # Per-slice statistics deliberately survive: inter-slice contrast is kept.
    slice_means = processed.reshape(processed.shape[0], -1).mean(dim=1)
    assert slice_means.abs().max() > 0.1


def test_medslim_per_slice_zscore_removes_inter_slice_contrast():
    """MedSliM per-slice z-score (the official cache does not do this)."""
    volume = _volume_with_slice_offsets()
    subject = tio.Subject(
        source=tio.ScalarImage(tensor=volume[None].clone(), affine=np.eye(4))
    )
    per_slice = tio.Compose([PerSliceZScore()])(subject)["source"].data
    for index in range(volume.shape[-1]):
        assert abs(float(per_slice[0, :, :, index].mean())) < 1e-4


def test_preprocess_curia_volume():
    volume_hwd = torch.randn(7, 5, 3)
    volume = volume_hwd.permute(1, 0, 2)[None]  # (1, W, H, D)
    pixel_values = preprocess_curia_volume(volume, crop_size=16)
    expected = _official_curia_preprocess(volume_hwd, crop_size=16)
    assert pixel_values.shape == (3, 1, 16, 16)
    torch.testing.assert_close(pixel_values, expected, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_preprocess_curia_volume_on_cuda_matches_cpu():
    volume_hwd = torch.randn(7, 5, 3)
    volume = volume_hwd.permute(1, 0, 2)[None]
    cpu = preprocess_curia_volume(volume, crop_size=16)
    gpu = preprocess_curia_volume(volume, crop_size=16, device=torch.device("cuda"))
    assert gpu.device.type == "cuda"
    torch.testing.assert_close(gpu.cpu(), cpu, rtol=1e-4, atol=1e-4)


def test_preprocess_curia_volume_rejects_non_grayscale_volumes():
    with pytest.raises(ValueError, match=r"grayscale \[1, W, H, D\]"):
        preprocess_curia_volume(torch.randn(3, 5, 7, 3), crop_size=16)


def test_official_processor_matches_the_reference_pipeline():
    """Parity with the pinned Hub processor; skipped if it cannot be loaded."""
    from baselines.curia.feature_cache import load_curia_image_processor

    try:
        processor = load_curia_image_processor()
    except Exception as error:  # offline, gated, or missing trust_remote_code deps
        pytest.skip(f"Pinned Curia image processor unavailable: {error}")

    volume = _volume_with_slice_offsets(depth=3, size=8)
    pixel_values = preprocess_curia_volume(
        volume.permute(1, 0, 2)[None], crop_size=int(processor.crop_size)
    )
    expected = _official_curia_preprocess(volume, crop_size=int(processor.crop_size))
    torch.testing.assert_close(pixel_values, expected, rtol=1e-4, atol=1e-4)
    hub = processor(volume, return_tensors="pt")["pixel_values"]
    if hub.ndim == 5:
        hub = hub[0]
    torch.testing.assert_close(pixel_values, hub, rtol=1e-4, atol=1e-4)


# Token cache
def _write_cache(tmp_path, uids=("a", "b"), depths=(9, 6), center=3):
    cache_dir = tmp_path / "cache"
    for uid, depth in zip(uids, depths):
        generator = torch.Generator().manual_seed(len(uid) + depth)
        patch = torch.randn(depth, PATCHES, EMBED_DIM, generator=generator)
        indices = select_slice_indices(depth, center)
        write_curia_cache_entry(
            path=cache_dir / f"{uid}.safetensors",
            cls_per_slice=torch.randn(depth, EMBED_DIM, generator=generator),
            patch_mean_per_slice=patch.mean(dim=1),
            center_patch_tokens=patch[indices],
            center_slice_indices=indices,
            metadata={
                "uid": uid,
                "num_slices": depth,
                "center_num_slices": center,
                "embed_dim": EMBED_DIM,
                "patches_per_slice": PATCHES,
            },
        )
    return cache_dir


def test_cache_dir_and_manifest_guard_preprocessing(tmp_path):
    path = curia_cache_dir("/hpcwork/rwth1833/feat_caches/curia_official", "kneeMRI", "train", "sagittal")
    assert path.as_posix() == "/hpcwork/rwth1833/feat_caches/curia_official/kneeMRI/train/sagittal"

    expected = cache_manifest(512, 3, "abcdef1234567890")
    write_cache_manifest(tmp_path, expected)
    assert_cache_manifest_compatible(tmp_path, expected)
    with pytest.raises(ValueError, match="different preprocessing"):
        assert_cache_manifest_compatible(tmp_path, cache_manifest(512, 3, "otherrev"))


def test_cache_roundtrip_reads_only_requested_tensors(tmp_path):
    cache_dir = _write_cache(tmp_path, uids=("a",), depths=(9,))
    tensors, metadata = read_curia_cache_entry(cache_dir / "a.safetensors")
    assert tensors[CLS_PER_SLICE].shape == (9, EMBED_DIM)
    assert tensors[PATCH_MEAN_PER_SLICE].shape == (9, EMBED_DIM)
    assert tensors[CENTER_PATCH_TOKENS].shape == (3, PATCHES, EMBED_DIM)
    assert metadata["num_slices"] == "9"

    subset, _ = read_curia_cache_entry(cache_dir / "a.safetensors", keys=[CLS_PER_SLICE])
    assert list(subset) == [CLS_PER_SLICE]
    with pytest.raises(KeyError, match="missing"):
        read_curia_cache_entry(cache_dir / "a.safetensors", keys=["nope"])


def test_cache_dataset_serves_raw_center_patches(tmp_path):
    cache_dir = _write_cache(tmp_path)
    dataset = CuriaTokenCacheDataset(cache_dir, token_source=["cls", "patch"], num_slices=3)
    item = dataset[0]
    assert item["patch_per_slice"].shape == (3, PATCHES, EMBED_DIM)
    assert item["cls_per_slice"].shape == (3, EMBED_DIM)

    cached, _ = read_curia_cache_entry(dataset.cache_path(item["uid"]))
    indices = select_slice_indices(cached[CLS_PER_SLICE].shape[0], 3)
    torch.testing.assert_close(item["cls_per_slice"], cached[CLS_PER_SLICE][indices])


def test_cache_dataset_serves_per_slice_means_as_single_patch_tokens(tmp_path):
    cache_dir = _write_cache(tmp_path)
    dataset = CuriaTokenCacheDataset(
        cache_dir, token_source=["patch"], use_avgpool_per_slice=True, num_slices=None
    )
    item = dataset[0]
    assert item["patch_per_slice"].shape == (9, 1, EMBED_DIM)
    assert "cls_per_slice" not in item


def test_cache_dataset_rejects_raw_patches_over_the_full_stack(tmp_path):
    cache_dir = _write_cache(tmp_path)
    with pytest.raises(ValueError, match="not cached"):
        CuriaTokenCacheDataset(cache_dir, token_source=["patch"], num_slices=None)


def test_cache_dataset_rejects_a_center_window_it_was_not_built_for(tmp_path):
    cache_dir = _write_cache(tmp_path, center=3)
    dataset = CuriaTokenCacheDataset(cache_dir, token_source=["patch"], num_slices=5)
    with pytest.raises(ValueError, match="centre window"):
        dataset[0]


def test_collate_pads_variable_depth_and_builds_the_slice_mask(tmp_path):
    cache_dir = _write_cache(tmp_path, depths=(9, 6))
    dataset = CuriaTokenCacheDataset(
        cache_dir, token_source=["cls"], num_slices=None
    )
    batch = curia_token_collate_fn([dataset[0], dataset[1]])
    assert batch["cls_per_slice"].shape == (2, 9, EMBED_DIM)
    assert batch["slice_mask"].sum(dim=1).tolist() == [9, 6]
    assert torch.count_nonzero(batch["cls_per_slice"][1, 6:]) == 0
    assert batch["uid"] == dataset.sample_ids


# Attention pooling
def test_attention_blocks_validation():
    assert validate_attention_blocks(["self", "cross"]) == ("self", "cross")
    assert validate_attention_blocks(["cross"]) == ("cross",)
    with pytest.raises(ValueError, match="Unsupported attention_blocks"):
        validate_attention_blocks(["self"])
    with pytest.raises(ValueError, match="Unknown attention block"):
        validate_attention_blocks(["self", "mlp"])
    with pytest.raises(ValueError, match="empty"):
        validate_attention_blocks([])
    with pytest.raises(ValueError, match="must be a list"):
        validate_attention_blocks("cross")
    with pytest.raises(ValueError, match="repeat"):
        validate_attention_blocks(["self", "self"])
    with pytest.raises(ValueError, match="repeat"):
        validate_attention_blocks(["cross", "cross"])
    with pytest.raises(ValueError, match="Unsupported attention_blocks"):
        validate_attention_blocks(["cross", "self"])


def test_cross_only_aggregator_has_no_self_attention_weights():
    default = InterSliceAggregator(embed_dim=EMBED_DIM)
    assert default.attention_blocks == ("cross",)
    assert not hasattr(default, "self_attention")
    cross_only = InterSliceAggregator(embed_dim=EMBED_DIM, attention_blocks=["cross"])
    assert not hasattr(cross_only, "self_attention")
    both = InterSliceAggregator(embed_dim=EMBED_DIM, attention_blocks=["self", "cross"])
    assert hasattr(both, "self_attention")
    assert sum(p.numel() for p in both.parameters()) > sum(
        p.numel() for p in cross_only.parameters()
    )


@pytest.mark.parametrize("blocks", [["cross"], ["self", "cross"]])
def test_aggregator_pools_to_one_embedding_per_volume(blocks):
    aggregator = InterSliceAggregator(
        embed_dim=EMBED_DIM, num_heads=1, num_queries=2, attention_blocks=blocks
    ).eval()
    tokens = torch.randn(3, 7, EMBED_DIM)
    pooled, attn = aggregator(tokens, return_attention=True)
    assert pooled.shape == (3, EMBED_DIM)
    assert attn.shape == (3, 2, 7)


@pytest.mark.parametrize("blocks", [["cross"], ["self", "cross"]])
def test_aggregator_ignores_padded_tokens(blocks):
    torch.manual_seed(0)
    aggregator = InterSliceAggregator(embed_dim=EMBED_DIM, attention_blocks=blocks).eval()
    tokens = torch.randn(2, 6, EMBED_DIM)
    mask = torch.tensor([[True] * 6, [True] * 3 + [False] * 3])

    with torch.no_grad():
        baseline = aggregator(tokens, mask=mask)
        perturbed_tokens = tokens.clone()
        perturbed_tokens[1, 3:] = 42.0
        perturbed = aggregator(perturbed_tokens, mask=mask)
    torch.testing.assert_close(baseline, perturbed)


def test_aggregator_train_path_does_not_materialize_attention_weights():
    aggregator = InterSliceAggregator(embed_dim=EMBED_DIM, attention_blocks=["self", "cross"])
    tokens = torch.randn(2, 5, EMBED_DIM)
    _, weights = aggregator.self_attention(tokens, tokens, need_weights=False)
    assert weights is None
    assert isinstance(aggregator(tokens), torch.Tensor)


def test_self_attention_is_not_a_noop_before_cross_attention():
    torch.manual_seed(0)
    tokens = torch.randn(2, 5, EMBED_DIM)
    cross_only = InterSliceAggregator(embed_dim=EMBED_DIM, attention_blocks=["cross"]).eval()
    both = InterSliceAggregator(embed_dim=EMBED_DIM, attention_blocks=["self", "cross"]).eval()
    both.cross_attention.load_state_dict(cross_only.cross_attention.state_dict())
    both.learned_queries.data.copy_(cross_only.learned_queries.data)
    with torch.no_grad():
        assert not torch.allclose(cross_only(tokens), both(tokens))


# Classifier trains from the cache without the backbone
@pytest.mark.parametrize(
    "recipe",
    [
        {"token_source": ["patch"], "num_slices": 3},
        {"token_source": ["patch"], "use_avgpool_per_slice": True},
        {"token_source": ["cls"]},
        {"token_source": ["cls", "patch"], "use_avgpool_per_slice": True},
    ],
)
def test_classifier_trains_one_step_from_cached_tokens(tmp_path, recipe):
    recipe = dict(recipe)
    num_slices = recipe.pop("num_slices", None)
    cache_dir = _write_cache(tmp_path)
    dataset = CuriaTokenCacheDataset(cache_dir, num_slices=num_slices, **recipe)
    batch = curia_token_collate_fn([dataset[0], dataset[1]])
    batch["labels"] = torch.tensor([0, 1])

    model = CuriaClassifier(embed_dim=EMBED_DIM, num_classes=2, **recipe)
    assert not any("backbone" in name for name, _ in model.named_parameters())
    assert sum(p.numel() for p in model.parameters()) < 10_000

    out = model(**_model_inputs(batch))
    assert out["logits"].shape == (2, 1)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        out["logits"].squeeze(-1), batch["labels"].float()
    )
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
        if p.requires_grad
    )


def test_classifier_linear_head_matches_the_official_shape():
    model = CuriaClassifier(embed_dim=EMBED_DIM, num_classes=3, token_source=["cls"])
    assert isinstance(model.classifier, torch.nn.Linear)
    assert model.classifier.out_features == 3
    legacy = CuriaClassifier(
        embed_dim=EMBED_DIM, num_classes=2, token_source=["cls"], classifier="mlp"
    )
    assert isinstance(legacy.classifier, torch.nn.Sequential)
    with pytest.raises(ValueError, match="Unknown classifier"):
        CuriaClassifier(embed_dim=EMBED_DIM, token_source=["cls"], classifier="attention")


def test_classifier_returns_attention_weights_on_request(tmp_path):
    cache_dir = _write_cache(tmp_path)
    dataset = CuriaTokenCacheDataset(cache_dir, token_source=["cls"], num_slices=None)
    batch = curia_token_collate_fn([dataset[0], dataset[1]])
    model = CuriaClassifier(embed_dim=EMBED_DIM, token_source=["cls"], num_queries=1).eval()
    with torch.no_grad():
        out = model(**_model_inputs(batch), return_attention=True)
    assert out["attn_weights"].shape == (2, 1, batch["cls_per_slice"].shape[1])
    # Padded slices get no attention mass.
    assert float(out["attn_weights"][1, 0, 6:].abs().sum()) == pytest.approx(0.0, abs=1e-6)


def test_classifier_rejects_invalid_token_source():
    with pytest.raises(ValueError, match="token_source"):
        CuriaClassifier(embed_dim=EMBED_DIM, token_source=["cls_patch"])


def test_binary_head_emits_a_single_logit():
    """Binary head is one logit, not two."""
    model = CuriaClassifier(embed_dim=EMBED_DIM, num_classes=2, token_source=["cls"])
    tokens = torch.randn(4, 5, EMBED_DIM)
    out = model(cls_per_slice=tokens, slice_mask=torch.ones(4, 5, dtype=torch.bool))
    assert out["logits"].shape == (4, 1)
    assert np.isfinite(out["logits"].detach().numpy()).all()


def test_labeled_dataset_follows_annotation_csv_order(tmp_path):
    cache_dir = _write_cache(tmp_path, uids=("z", "a", "m"), depths=(6, 6, 6))
    csv_path = tmp_path / "train.csv"
    csv_path.write_text("ID,acl\na,0\nm,1\nz,2\n")
    dataset = LabeledCuriaTokenDataset(
        cache_dir, str(csv_path), "multiclass", ["acl"], token_source=["cls"]
    )
    assert dataset.sample_ids == ["a", "m", "z"]


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
        assert set(train_a).isdisjoint(val_a)
        assert set(train_a).union(val_a) == set(range(len(labels)))


def test_train_val_split_is_stratified_and_seeded():
    labels = np.array([0] * 10 + [1] * 10 + [2] * 10)
    seed = _derive_fold_seed(42)
    train_idx, val_idx = _train_val_indices(labels, 0.1, "multiclass", seed)
    train_b, val_b = _train_val_indices(labels, 0.1, "multiclass", seed)
    np.testing.assert_array_equal(train_idx, train_b)
    np.testing.assert_array_equal(val_idx, val_b)
    assert set(train_idx).isdisjoint(val_idx)
    assert set(train_idx).union(val_idx) == set(range(len(labels)))
