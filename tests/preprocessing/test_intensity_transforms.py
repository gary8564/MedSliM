"""
Tests for model-specific slice intensity preprocessing policies.

Covers:
- The reusable intensity transforms (ClipIntensity, PerSliceZScore).
- Model-name policy resolution in transforms._build_intensity_transforms:
  * med-dinov3: CT clamp [-1000, 1000] + fixed CT mean/std z-score.
  * flexict-2d: CT clamp [-1000, 1000] + per-slice z-score.
  * curia: modality-aware CT air clipping + per-slice z-score.
  * other models: fixed image_mean/image_std z-score.
- CLI modality resolution/validation (resolve_modality).
"""

import logging

import numpy as np
import pytest
import torch
import torchio as tio

from med_slim.utils.model_config import get_slice_encoder_config
from med_slim.utils.preprocessing.augmentation import (
    ClipIntensity,
    PerSliceZScore,
)
from med_slim.utils.preprocessing.transforms import (
    CT_HU_MAX,
    CT_HU_MIN,
    _build_intensity_transforms,
)
from med_slim.utils.preprocessing.precompute_slice_feature import resolve_modality

MED_DINOV3_MEAN = 65.1084213256836
MED_DINOV3_STD = 178.01663208007812
_MASKING = lambda x: (x > x.min()) & (x < x.max())


def _subject(data: np.ndarray) -> tio.Subject:
    """Wrap a (C, W, H, D) array in a TorchIO subject with identity affine."""
    return tio.Subject(source=tio.ScalarImage(tensor=torch.from_numpy(data), affine=np.eye(4)))


def _apply(transforms: list, data: np.ndarray) -> torch.Tensor:
    subject = tio.Compose(transforms)(_subject(data))
    return subject["source"].data


def test_clip_intensity_clamps_both_bounds():
    data = np.array([[-2000.0, -1000.0, 0.0, 1000.0, 2000.0]], dtype=np.float32).reshape(1, 5, 1, 1)
    out = _apply([ClipIntensity(min_value=-1000.0, max_value=1000.0)], data)
    assert out.min().item() == -1000.0
    assert out.max().item() == 1000.0


def test_clip_intensity_requires_a_bound():
    with pytest.raises(ValueError):
        ClipIntensity()


def test_clip_intensity_below_air_enabled_raises_floor():
    data = np.array([[-3000.0, -1000.0, 500.0]], dtype=np.float32).reshape(1, 3, 1, 1)
    out = _apply([ClipIntensity(min_value=-1000.0, enabled=True)], data)
    assert out.min().item() == -1000.0
    # Values above air are untouched (no upper clamp).
    assert out.max().item() == 500.0


def test_clip_intensity_disabled_is_noop():
    data = np.array([[-3000.0, -1000.0, 500.0]], dtype=np.float32).reshape(1, 3, 1, 1)
    out = _apply([ClipIntensity(min_value=-1000.0, enabled=False)], data)
    assert out.min().item() == -3000.0


def test_per_slice_zscore_is_independent_per_slice():
    rng = np.random.default_rng(0)
    # Two slices with very different offsets/scales. After z-scoring, each should end ~N(0, 1).
    data = np.stack(
        [rng.normal(100.0, 5.0, size=(8, 8)), rng.normal(-50.0, 20.0, size=(8, 8))],
        axis=-1,
    ).astype(np.float32)[None]  # (1, 8, 8, 2)
    out = _apply([PerSliceZScore()], data)
    for d in range(2):
        sl = out[0, :, :, d]
        assert abs(float(sl.mean())) < 1e-4
        # Unbiased std as in Curia's processor.
        assert abs(float(sl.std()) - 1.0) < 1e-2


def test_per_slice_zscore_empty_slice_no_nan():
    data = np.zeros((1, 4, 4, 1), dtype=np.float32)
    out = _apply([PerSliceZScore()], data)
    assert torch.isfinite(out).all()


def test_med_dinov3_transforms():
    cfg = get_slice_encoder_config("med-dinov3")
    means, stds = list(cfg["image_mean"]), list(cfg["image_std"])
    transforms = _build_intensity_transforms("med-dinov3", "ct", means, stds, _MASKING)

    data = np.array([-2000.0, -1000.0, 0.0, 65.1084213256836, 1000.0, 3000.0], dtype=np.float32)
    data = data.reshape(1, 6, 1, 1)
    out = _apply(transforms, data).numpy()

    expected = (np.clip(data, CT_HU_MIN, CT_HU_MAX) - MED_DINOV3_MEAN) / MED_DINOV3_STD
    np.testing.assert_allclose(out, expected, rtol=0, atol=1e-4)


def test_med_dinov3_config_uses_fixed_stats():
    cfg = get_slice_encoder_config("med-dinov3")
    assert cfg["image_mean"][0] == pytest.approx(MED_DINOV3_MEAN)
    assert cfg["image_std"][0] == pytest.approx(MED_DINOV3_STD)


def test_flexict_transforms():
    cfg = get_slice_encoder_config("flexict-2d")
    means, stds = list(cfg["image_mean"]), list(cfg["image_std"])
    transforms = _build_intensity_transforms("flexict-2d", "ct", means, stds, _MASKING)

    rng = np.random.default_rng(1)
    data = rng.normal(200.0, 400.0, size=(1, 8, 8, 3)).astype(np.float32)
    data[0, 0, 0, 0] = 5000.0   # above HU max -> clamped
    data[0, 1, 1, 0] = -5000.0  # below HU min -> clamped
    out = _apply(transforms, data)

    # Per-slice standardized (approx zero mean / unit std).
    for d in range(3):
        sl = out[0, :, :, d]
        assert abs(float(sl.mean())) < 1e-4
        assert abs(float(sl.std()) - 1.0) < 1e-2


def test_flexict_image_size_targets_512():
    cfg = get_slice_encoder_config("flexict-2d")
    assert tuple(cfg["img_size"]) == (512, 512)


def test_curia_transforms():
    cfg = get_slice_encoder_config("curia")
    means, stds = list(cfg["image_mean"]), list(cfg["image_std"])
    transforms = _build_intensity_transforms("curia", "ct", means, stds, _MASKING)

    assert len(transforms) == 2
    assert isinstance(transforms[0], ClipIntensity)
    assert transforms[0].enabled is True
    assert transforms[0].min_value == CT_HU_MIN
    assert transforms[0].max_value is None
    assert isinstance(transforms[1], PerSliceZScore)

    data = np.array([-3000.0, -500.0, 800.0, 400.0], dtype=np.float32).reshape(1, 4, 1, 1)
    out = _apply(transforms, data)
    expected_clipped = np.clip(data, CT_HU_MIN, None)
    expected = (expected_clipped - expected_clipped.mean(axis=(1, 2), keepdims=True)) / expected_clipped.std(axis=(1, 2), ddof=1, keepdims=True)
    np.testing.assert_allclose(out.numpy(), expected, rtol=0, atol=1e-5)


def test_curia_mri_skips_air_clip():
    cfg = get_slice_encoder_config("curia")
    means, stds = list(cfg["image_mean"]), list(cfg["image_std"])
    transforms = _build_intensity_transforms("curia", "mri", means, stds, _MASKING)

    assert len(transforms) == 2
    assert isinstance(transforms[0], ClipIntensity)
    assert transforms[0].enabled is False
    assert transforms[0].min_value == CT_HU_MIN
    assert transforms[0].max_value is None
    assert isinstance(transforms[1], PerSliceZScore)

    data = np.array([-3000.0, -500.0, 800.0, 400.0], dtype=np.float32).reshape(1, 4, 1, 1)
    out = _apply(transforms, data)
    expected = (data - data.mean(axis=(1, 2), keepdims=True)) / data.std(axis=(1, 2), ddof=1, keepdims=True)
    np.testing.assert_allclose(out.numpy(), expected, rtol=0, atol=1e-5)


def test_other_model_uses_fixed_stats():
    cfg = get_slice_encoder_config("dinov2")
    means, stds = list(cfg["image_mean"]), list(cfg["image_std"])
    transforms = _build_intensity_transforms("dinov2", "mri", means, stds, _MASKING)

    # Legacy path is a single ZNormalization with the config stats.
    from med_slim.utils.preprocessing.augmentation import ZNormalization

    assert len(transforms) == 1
    assert isinstance(transforms[0], ZNormalization)
    assert transforms[0].channelwise_precomputed_means == means
    assert transforms[0].channelwise_precomputed_stds == stds


@pytest.mark.parametrize("model", ["med-dinov3", "flexict-2d"])
def test_resolve_modality_ct_only_models_defaults_to_none(model):
    assert resolve_modality(model, None) is None
    assert resolve_modality(model, "ct") == "ct"


@pytest.mark.parametrize("model", ["med-dinov3", "flexict-2d"])
def test_resolve_modality_ct_only_models_warns_when_passing_mri(model, caplog):
    with caplog.at_level(logging.WARNING):
        result = resolve_modality(model, "mri")
    assert result == "mri"
    assert "has no effect" in caplog.text


def test_resolve_modality_mri_only_models_defaults_to_none():
    assert resolve_modality("mri-core", None) is None
    assert resolve_modality("mri-core", "mri") == "mri"


def test_resolve_modality_mri_only_models_warns_when_passing_ct(caplog):
    with caplog.at_level(logging.WARNING):
        result = resolve_modality("mri-core", "ct")
    assert result == "ct"
    assert "has no effect" in caplog.text


def test_resolve_modality_curia_requires_explicit_value():
    with pytest.raises(ValueError):
        resolve_modality("curia", None)
    assert resolve_modality("curia", "ct") == "ct"
    assert resolve_modality("curia", "mri") == "mri"


def test_resolve_modality_other_models_returns_none_by_default():
    assert resolve_modality("dinov2", None) is None


def test_resolve_modality_other_models_returns_explicit_value_when_set():
    assert resolve_modality("dinov2", "ct") == "ct"
