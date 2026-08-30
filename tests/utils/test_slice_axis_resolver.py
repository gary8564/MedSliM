"""Tests for slice-axis resolution and preprocessing pipeline integration."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torchio as tio

from med_slim.utils.preprocessing.augmentation import (
    EnsureSliceAxisLast,
    _slice_axis_from_subject,
)
from med_slim.utils.preprocessing.slice_axis_resolver import (
    PLANE_NORMALS_RAS,
    build_slice_last_affine,
    geometric_slice_axis,
    resolve_slice_axis,
    slice_normal_from_affine,
    unit_vector,
)
from med_slim.utils.preprocessing.transforms import get_adaptive_transform


def _subject(shape: tuple[int, int, int], spacing: tuple[float, float, float]) -> tio.Subject:
    data = np.zeros((1, *shape), dtype=np.float32)
    return tio.Subject(image=tio.ScalarImage(tensor=data, affine=np.diag([*spacing, 1.0])))


def _subject_with_affine(shape: tuple[int, int, int], affine: np.ndarray) -> tio.Subject:
    data = np.zeros((1, *shape), dtype=np.float32)
    return tio.Subject(image=tio.ScalarImage(tensor=data, affine=affine))


@dataclass(frozen=True)
class VolumeCase:
    """One preprocessed NIfTI per dataset (paths from preprocess_*.py defaults)."""

    dataset: str
    path: Path
    plane: str
    expected_slices: int


# Default save-dir layout from scripts/preprocess_dataset/preprocess_*.py
VOLUME_CASES = [
    VolumeCase(
        "fastMRI",
        Path(
            "/hpcwork/rwth1833/datasets/preprocessed/fastMRI/train/pd_fs/coronal/"
            "study_80d2ab0b_MR6_8d1bff90.nii.gz"
        ),
        "coronal",
        35,
    ),
    VolumeCase(
        "MRNet",
        Path("/hpcwork/rwth1833/datasets/preprocessed/MRNet/test/sagittal/1206.nii.gz"),
        "sagittal",
        32,
    ),
    VolumeCase(
        "KMAR-50K",
        Path(
            "/hpcwork/rwth1833/datasets/preprocessed/KMAR-50K/train/axial/"
            "2020_291_MR0.nii.gz"
        ),
        "axial",
        20,
    ),
    VolumeCase(
        "kneeMRI",
        Path(
            "/hpcwork/rwth1833/datasets/preprocessed/kneeMRI/test/sagittal/"
            "491596-5.nii.gz"
        ),
        "sagittal",
        26,
    ),
    VolumeCase(
        "RSNA-Knee-2D-axial",
        Path(
            "/hpcwork/rwth1833/datasets/preprocessed/RSNA-Knee/test/non_fluid_sensitive/axial/"
            "1.2.826.0.1.3680043.8.498.10047035057544427318018579121635276191_"
            "MR0_1.2.826.0.1.3680043.8.498.11580656442259111255675562605155903947"
            ".nii.gz"
        ),
        "axial",
        34,
    ),
    VolumeCase(
        "RSNA-Knee-3D-sagittal",
        Path(
            "/hpcwork/rwth1833/datasets/preprocessed/RSNA-Knee/train/fluid_sensitive_fs/sagittal/"
            "1.2.826.0.1.3680043.8.498.10085187975213640798228717119613397941_"
            "MR1_1.2.826.0.1.3680043.8.498.30612781365806999657709742696576857451"
            ".nii.gz"
        ),
        "sagittal",
        320,
    ),
    VolumeCase(
        "SKM-TEA",
        Path(
            "/hpcwork/rwth1833/datasets/preprocessed/SKM-TEA/DESS_E1/train/sagittal/"
            "MTR_095.nii.gz"
        ),
        "sagittal",
        160,
    ),
    VolumeCase(
        "LIDC",
        Path("/hpcwork/rwth1833/datasets/preprocessed/LIDC-IDRI/train/axial/159.nii.gz"),
        "axial",
        251,
    ),
]


@pytest.fixture
def real_volume(request: pytest.FixtureRequest) -> VolumeCase:
    case: VolumeCase = request.param
    if not case.path.is_file():
        pytest.skip(f"Missing {case.dataset} sample: {case.path}")
    return case


class TestAffineHelpers:
    def test_unit_vector_normalizes(self):
        np.testing.assert_allclose(unit_vector([0.0, 0.0, 3.0]), [0.0, 0.0, 1.0])

    def test_unit_vector_rejects_degenerate_inputs(self):
        assert not unit_vector([0.0, 0.0, 0.0]).any()
        assert not unit_vector([np.nan, 1.0, 0.0]).any()
        assert not unit_vector([1.0, 2.0]).any()

    def test_build_slice_last_affine(self):
        affine = build_slice_last_affine("sagittal")
        np.testing.assert_allclose(
            slice_normal_from_affine(affine, 2), PLANE_NORMALS_RAS["sagittal"]
        )

    def test_slice_normal_from_affine(self):
        affine = np.eye(4)
        affine[:3, 2] = [0.0, 4.0, 0.0]
        np.testing.assert_allclose(slice_normal_from_affine(affine, 2), [0.0, 1.0, 0.0])


class TestGeometricSliceAxis:
    def test_spacing_selects_thickest_axis(self):
        assert geometric_slice_axis((512, 512, 600), (0.7, 0.7, 1.0), plane="axial") == 2

    def test_shape_used_when_smallest_axis_matches_plane(self):
        assert geometric_slice_axis((33, 320, 320), (1.0, 1.0, 1.0), plane="sagittal") == 0

    def test_smallest_axis_conflicts_with_plane(self):
        assert geometric_slice_axis((256, 256, 30), (1.0, 1.0, 1.0), plane="sagittal") is None
        assert geometric_slice_axis((320, 320, 26), (1.0, 1.0, 1.0), plane="sagittal") is None

    def test_isotropic_cube(self):
        assert geometric_slice_axis((256, 256, 256), (1.0, 1.0, 1.0), plane="axial") is None

    @pytest.mark.parametrize(
        "shape, plane, expected_plane_axis",
        [
            ((512, 600, 1200), "axial", 2),
            ((1200, 512, 600), "sagittal", 0),
            ((512, 1200, 600), "coronal", 1),
        ],
    )
    def test_abstains_on_rectangular_isotropic_ct(self, shape, plane, expected_plane_axis):
        """Rectangular in-plane FOV must not be mistaken for the slice axis."""
        assert geometric_slice_axis(shape, (0.5, 0.5, 0.5), plane=plane) is None
        assert resolve_slice_axis(shape, (0.5, 0.5, 0.5), plane=plane) == expected_plane_axis


class TestResolveSliceAxis:
    def test_spacing_wins_over_plane_label(self):
        assert resolve_slice_axis((36, 320, 320), (3.3, 0.47, 0.47), plane="coronal") == 0

    def test_coronal_MRI_spacing(self):
        assert resolve_slice_axis((320, 35, 320), (0.47, 3.3, 0.47), plane="coronal") == 1

    def test_sagittal_MRI_spacing(self):
        assert resolve_slice_axis((320, 33, 320), (0.47, 3.3, 0.47), plane="sagittal") == 1

    def test_ct_anisotropic_spacing(self):
        assert resolve_slice_axis((512, 512, 600), (0.7, 0.7, 1.0), plane="axial") == 2

    def test_affine_resolves_isotropic_cube(self):
        affine = build_slice_last_affine("axial")
        assert resolve_slice_axis(
            (256, 256, 256), (1.0, 1.0, 1.0), plane="axial", affine=affine
        ) == 2

    def test_fallback_when_geometry_unresolved(self):
        assert resolve_slice_axis((256, 256, 256), (1.0, 1.0, 1.0), plane="coronal") == 1

    def test_fallback_when_affine_unresolved(self):
        assert resolve_slice_axis((256, 256, 30), (1.0, 1.0, 1.0), plane="sagittal") == 0
        assert resolve_slice_axis(
            (256, 256, 30),
            (1.0, 1.0, 1.0),
            plane="sagittal",
            affine=build_slice_last_affine("sagittal"),
        ) == 2

    def test_fallback_when_affine_resolved(self):
        affine = build_slice_last_affine("sagittal")
        assert resolve_slice_axis(
            (256, 256, 256), (1.0, 1.0, 1.0), plane="sagittal", affine=affine
        ) == 2

    def test_high_res_isotropic_ct(self):
        assert resolve_slice_axis((1024, 1024, 800), (1.0, 1.0, 1.0), plane="axial") == 2

    def test_rejects_unknown_plane(self):
        with pytest.raises(ValueError, match="Unknown plane"):
            resolve_slice_axis((256, 256, 256), (1.0, 1.0, 1.0), plane="oblique")


class TestSubjectSliceAxis:
    def test_fastmri_coronal_spacing_on_axis0(self):
        assert _slice_axis_from_subject(_subject((36, 320, 320), (3.3, 0.47, 0.47)), "coronal") == 0

    def test_fastmri_sagittal_spacing_on_axis1(self):
        assert _slice_axis_from_subject(_subject((320, 33, 320), (0.47, 3.3, 0.47)), "sagittal") == 1

    def test_coronal_CT_spacing(self):
        assert _slice_axis_from_subject(_subject((512, 30, 512), (0.31, 4.0, 0.31)), "coronal") == 1

    def test_ct_axial_anisotropic_spacing(self):
        assert _slice_axis_from_subject(_subject((512, 512, 384), (0.70, 0.70, 1.00)), "axial") == 2

    def test_fallback_when_isotropic_ct(self):
        assert _slice_axis_from_subject(_subject((512, 512, 480), (0.70, 0.70, 0.70)), "axial") == 2
        assert _slice_axis_from_subject(_subject((1024, 1024, 800), (0.5, 0.5, 0.5)), "axial") == 2

    def test_fallback_when_affine_resolved(self):
        subj = _subject_with_affine((256, 256, 30), build_slice_last_affine("sagittal"))
        assert _slice_axis_from_subject(subj, "sagittal") == 2


class TestEnsureSliceAxisLast:
    def test_places_slices_on_last_axis(self):
        # Caase 1
        data_1 = np.zeros((1, 512, 512, 600), dtype=np.float32)
        affine_1 = np.diag([0.7, 0.7, 1.0, 1.0])
        subject_1 = tio.Subject(source=tio.ScalarImage(tensor=data_1, affine=affine_1))
        out_1 = tio.Compose([tio.ToCanonical(), EnsureSliceAxisLast(plane="axial")])(subject_1)
        assert out_1["source"].spatial_shape[2] == 600

        # Case 2
        data_2 = np.zeros((1, 36, 320, 320), dtype=np.float32)
        affine_2 = np.diag([3.3, 0.47, 0.47, 1.0])
        subject_2 = tio.Subject(source=tio.ScalarImage(tensor=data_2, affine=affine_2))
        out_2 = tio.Compose([tio.ToCanonical(), EnsureSliceAxisLast(plane="coronal")])(subject_2)
        assert out_2["source"].spatial_shape[2] == 36


class TestVolumeSliceAxis:
    @pytest.mark.parametrize(
        "real_volume",
        VOLUME_CASES,
        indirect=True,
        ids=[case.dataset for case in VOLUME_CASES],
    )
    def test_esure_slice_last_after_canonical(self, real_volume: VolumeCase):
        img = tio.ToCanonical()(tio.ScalarImage(real_volume.path))
        transformed = tio.Compose([EnsureSliceAxisLast(plane=real_volume.plane)])(
            tio.Subject(source=img)
        )
        assert transformed["source"].spatial_shape[2] == real_volume.expected_slices

    @pytest.mark.parametrize(
        "real_volume",
        VOLUME_CASES,
        indirect=True,
        ids=[case.dataset for case in VOLUME_CASES],
    )
    def test_subject_slice_axis_matches_volume_shape(self, real_volume: VolumeCase):
        img = tio.ToCanonical()(tio.ScalarImage(real_volume.path))
        subj = tio.Subject(image=img)
        axis = _slice_axis_from_subject(subj, real_volume.plane)
        assert subj.image.spatial_shape[axis] == real_volume.expected_slices


class TestVolumeAdaptiveTransform:
    @pytest.mark.parametrize(
        "real_volume",
        VOLUME_CASES,
        indirect=True,
        ids=[case.dataset for case in VOLUME_CASES],
    )
    def test_adaptive_transform_slice_count(self, real_volume: VolumeCase):
        transform = get_adaptive_transform(
            model_name="dinov2", plane=real_volume.plane, num_slices=None
        )
        out = transform(tio.ScalarImage(real_volume.path))
        assert out.shape[1] == real_volume.expected_slices, (
            f"{real_volume.dataset}: expected {real_volume.expected_slices} slices, "
            f"got {out.shape[1]}"
        )

    @pytest.mark.parametrize("model_name", ["dinov2", "curia", "medimageinsight"])
    def test_adaptive_transform_for_different_fms(self, model_name: str):
        case = VOLUME_CASES[0]
        if not case.path.is_file():
            pytest.skip(f"Missing fastMRI sample: {case.path}")

        transform = get_adaptive_transform(
            model_name=model_name, plane=case.plane, num_slices=None
        )
        out = transform(tio.ScalarImage(case.path))
        assert out.shape[1] == case.expected_slices, (
            f"{model_name}: expected {case.expected_slices} slices, got {out.shape[1]}"
        )
