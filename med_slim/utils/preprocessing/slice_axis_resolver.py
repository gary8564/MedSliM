"""
Slice-axis resolution for canonical (RAS+) 3D volumes.
"""
import numpy as np

from typing import Sequence

PLANE_NORMALS_RAS = {
    "sagittal": (1.0, 0.0, 0.0),
    "coronal": (0.0, 1.0, 0.0),
    "axial": (0.0, 0.0, 1.0),
}

PLANE_TO_AXIS = {"sagittal": 0, "coronal": 1, "axial": 2}

PLANE_INPLANE_AXES_RAS = {
    "sagittal": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    "coronal": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "axial": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
}


def unit_vector(vec: Sequence[float]) -> np.ndarray:
    """Return a finite unit vector, or zeros if the input is degenerate."""
    arr = np.asarray(vec, dtype=float).reshape(-1)
    if arr.size != 3 or not np.isfinite(arr).all():
        return np.zeros(3, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm <= 0:
        return np.zeros(3, dtype=float)
    return arr / norm


def build_slice_last_affine(
    plane: str,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """
    Build a 4x4 affine for slice-last TorchIO layout ``(C, W, H, D)``.

    ``D`` is the through-plane direction; its affine column aligns with
    ``PLANE_NORMALS_RAS[plane]``.
    """
    if plane not in PLANE_NORMALS_RAS:
        raise ValueError(
            f"Unknown plane '{plane}'. Expected one of {sorted(PLANE_NORMALS_RAS)}."
        )
    sp = np.asarray(spacing, dtype=float).reshape(3)
    if sp.size != 3 or not np.isfinite(sp).all() or (sp <= 0).any():
        raise ValueError(f"spacing must be three positive finite values, got {spacing!r}.")
    d0, d1 = PLANE_INPLANE_AXES_RAS[plane]
    normal = np.asarray(PLANE_NORMALS_RAS[plane], dtype=float)
    affine = np.eye(4)
    affine[:3, 0] = np.asarray(d0, dtype=float) * sp[0]
    affine[:3, 1] = np.asarray(d1, dtype=float) * sp[1]
    affine[:3, 2] = normal * sp[2]
    return affine


def slice_normal_from_affine(affine: np.ndarray, slice_axis: int) -> np.ndarray:
    """Physical (RAS) direction of `slice_axis` as a unit vector."""
    affine = np.asarray(affine, dtype=float)
    if affine.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 affine, got shape {affine.shape}.")
    if slice_axis not in (0, 1, 2):
        raise ValueError(f"slice_axis must be 0, 1 or 2, got {slice_axis}.")
    return unit_vector(affine[:3, slice_axis])


def _get_largest_spacing_axis(spacing: np.ndarray) -> int | None:
    """
    Return the unique axis with the largest voxel spacing.
    The through-plane axis is the one carrying the thickest slices.
    """
    if spacing.size != 3 or not np.isfinite(spacing).all() or (spacing <= 0).any():
        return None
    largest = float(spacing.max())
    axes = np.flatnonzero(np.isclose(spacing, largest))
    return int(axes[0]) if len(axes) == 1 else None


def _get_smallest_shape_axis(shape: np.ndarray) -> int | None:
    """Return the unique axis with the fewest voxels, or ``None`` if ambiguous."""
    if shape.size != 3 or not np.isfinite(shape).all() or (shape <= 0).any():
        return None
    smallest = float(shape.min())
    axes = np.flatnonzero(shape == smallest)
    return int(axes[0]) if len(axes) == 1 else None


def _smallest_shape_axis_matches_plane(shape_axis: int, plane: str) -> bool:
    """Accept shape evidence only when the smallest dimension sits on the plane's expected array index."""
    return shape_axis == PLANE_TO_AXIS[plane]


def _plane_axis_from_affine(affine: np.ndarray, plane: str) -> int | None:
    """Project the RAS orientation axis of the corresponding plane onto the affine to get the slice axis."""
    normal = np.asarray(PLANE_NORMALS_RAS[plane], dtype=float)
    affine = np.asarray(affine, dtype=float)
    if affine.shape != (4, 4) or not np.isfinite(affine[:3, :3]).all():
        return None

    scores = np.array(
        [abs(float(np.dot(unit_vector(affine[:3, axis]), normal))) for axis in range(3)],
        dtype=float,
    )
    if not np.isfinite(scores).any() or scores.max() <= 0:
        return None
    best_axes = np.flatnonzero(np.isclose(scores, scores.max()))
    return int(best_axes[0]) if len(best_axes) == 1 else None


def geometric_slice_axis(
    shape: Sequence[float], spacing: Sequence[float], plane: str
) -> int | None:
    """
    Geometry-only slice-axis estimate (spacing first, then shape).
    Returns None when spacing and shape are both uninformative.
    """
    if plane not in PLANE_TO_AXIS:
        raise ValueError(
            f"Unknown plane '{plane}'. Expected one of {sorted(PLANE_TO_AXIS)}."
        )
    shape = np.asarray(shape, dtype=float).reshape(-1)
    spacing = np.asarray(spacing, dtype=float).reshape(-1)

    spacing_axis = _get_largest_spacing_axis(spacing)
    if spacing_axis is not None:
        return spacing_axis

    shape_axis = _get_smallest_shape_axis(shape)
    if shape_axis is None:
        return None
    if _smallest_shape_axis_matches_plane(shape_axis, plane):
        return shape_axis
    return None


def resolve_slice_axis(
    shape: Sequence[float],
    spacing: Sequence[float],
    plane: str,
    affine: np.ndarray | None = None,
) -> int:
    """
    Resolve the through-plane array axis for a canonicalized volume.

    1. Geometry: unique thickest spacing, then plane-aligned thinnest shape.
    2. Affine: project `plane` onto the canonical affine.
    3. `PLANE_TO_AXIS[plane]`.
    """
    if plane not in PLANE_TO_AXIS:
        raise ValueError(
            f"Unknown plane '{plane}'. Expected one of {sorted(PLANE_TO_AXIS)}."
        )

    geo = geometric_slice_axis(shape, spacing, plane)
    if geo is not None:
        return geo

    if affine is not None:
        affine_axis = _plane_axis_from_affine(affine, plane)
        if affine_axis is not None:
            return affine_axis

    return PLANE_TO_AXIS[plane]
