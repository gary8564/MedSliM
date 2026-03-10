"""
Unit tests for the precompute_slice_feature pipeline.

Validates that:
- SliceDataset + transforms produce correct tensor shape (C, W, H, D)
- slice_collate_fn produces correct batch shape (B, C, W, H, D)
- Output shape matches slice encoder expectations
"""
import unittest
import tempfile
import numpy as np
import nibabel as nib
import pandas as pd
import torch
import torchio as tio
from pathlib import Path

from med_slim.data.slice_dataset import SliceDataset, slice_collate_fn
from med_slim.utils.preprocessing.transforms import get_transforms
from med_slim.utils.model_config import get_slice_encoder_config
from med_slim.model.slice_encoder import build_slice_encoder

def _make_synthetic_dataset(tmp_path: Path, num_samples: int = 2) -> Path:
    """Create a minimal synthetic MR-like dataset for shape testing."""
    # Volume shape: (W, H, D) in nibabel/RAS order; D=slice dimension (smallest)
    w, h, d = 320, 320, 22
    plane_dir = tmp_path / "train" / "axial"
    plane_dir.mkdir(parents=True)

    for i in range(num_samples):
        uid = f"{i:04d}"
        vol = np.random.randn(w, h, d).astype(np.float32)
        aff = np.eye(4)
        img = nib.Nifti1Image(vol, aff)
        nib.save(img, plane_dir / f"{uid}.nii.gz")

    df = pd.DataFrame({
        "ID": [f"{i:04d}" for i in range(num_samples)],
        "abnormal": [i % 2 for i in range(num_samples)],
        "acl": [0] * num_samples,
        "meniscus": [1 - (i % 2) for i in range(num_samples)],
    })
    df.to_csv(tmp_path / "train.csv", index=False)
    return tmp_path


class TestPrecomputeSliceFeatureShapes(unittest.TestCase):
    """Test that the precompute pipeline produces correct tensor shapes."""

    def setUp(self):
        self.model_name = "biomedclip"
        self.cfg = get_slice_encoder_config(self.model_name)
        self.H_crop, self.W_crop = tuple(self.cfg["img_size"])
        self.num_slices = 32

    def test_single_sample_shape_resample_mode(self):
        """Single sample: (C, W, H, D) with C=1, W=img_size[0], H=img_size[1], D=num_slices."""
        with self.subTest("create synthetic data"):
            with tempfile.TemporaryDirectory() as tmp:
                data_root = Path(tmp)
                _make_synthetic_dataset(data_root, num_samples=1)

                _, val_tf = get_transforms(
                    model_name=self.model_name,
                    num_slices=self.num_slices,
                    spatial_mode="resample",
                )
                ds = SliceDataset(
                    path_root=str(data_root),
                    split="train",
                    transform=val_tf,
                    plane="axial",
                )
                sample = ds[0]
                src = sample["source"]
                self.assertIsInstance(src, tio.ScalarImage)
                t = src.tensor
                self.assertEqual(t.ndim, 4, "Expected 4D (C, W, H, D)")
                C, W, H, D = t.shape
                self.assertEqual(C, 1, "Grayscale MRI: C=1")
                self.assertEqual(W, self.W_crop, f"W should match config img_size[0]={self.W_crop}")
                self.assertEqual(H, self.H_crop, f"H should match config img_size[1]={self.H_crop}")
                self.assertEqual(D, self.num_slices, f"D should be num_slices={self.num_slices}")

    def test_batch_shape_from_collate(self):
        """Collated batch: (B, C, W, H, D) with B>=1."""
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            _make_synthetic_dataset(data_root, num_samples=3)

            _, val_tf = get_transforms(
                model_name=self.model_name,
                num_slices=self.num_slices,
                spatial_mode="resample",
            )
            ds = SliceDataset(
                path_root=str(data_root),
                split="train",
                transform=val_tf,
                plane="axial",
            )
            dl = torch.utils.data.DataLoader(
                ds,
                batch_size=2,
                shuffle=False,
                collate_fn=slice_collate_fn,
            )
            batch = next(iter(dl))
            x = batch["x"]
            self.assertEqual(x.ndim, 5, "Batch should be 5D (B, C, W, H, D)")
            B, C, W, H, D = x.shape
            self.assertEqual(B, 2)
            self.assertEqual(C, 1)
            self.assertEqual(W, self.W_crop)
            self.assertEqual(H, self.H_crop)
            self.assertEqual(D, self.num_slices)

    def test_slice_encoder_input_compatibility(self):
        """Batch shape (B, C, W, H, D) must match slice encoder expectation."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            _make_synthetic_dataset(data_root, num_samples=1)

            _, val_tf = get_transforms(
                model_name=self.model_name,
                num_slices=self.num_slices,
                spatial_mode="resample",
            )
            ds = SliceDataset(
                path_root=str(data_root),
                split="train",
                transform=val_tf,
                plane="axial",
            )
            dl = torch.utils.data.DataLoader(
                ds,
                batch_size=1,
                shuffle=False,
                collate_fn=slice_collate_fn,
            )
            batch = next(iter(dl))
            x = batch["x"]
            # Slice encoder expects [B, C, W, H, D] and swaps to [B, C, D, H, W]
            self.assertEqual(
                x.shape,
                (1, 1, self.W_crop, self.H_crop, self.num_slices),
                f"Expected (1, 1, {self.W_crop}, {self.H_crop}, {self.num_slices})",
            )

    def test_raw_slice_resolution_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            _make_synthetic_dataset(data_root, num_samples=1)

            _, val_tf = get_transforms(
                model_name=self.model_name,
                num_slices=None,
                spatial_mode="resample",
            )
            ds = SliceDataset(
                path_root=str(data_root),
                split="train",
                transform=val_tf,
                plane="axial",
            )
            sample = ds[0]
            t = sample["source"].tensor
            C, W, H, D = t.shape
            self.assertEqual(C, 1)
            self.assertEqual(W, self.W_crop)
            self.assertEqual(H, self.H_crop)
            self.assertEqual(D, 22, "Raw mode keeps original 22 slices")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required for encoder test")
    def test_end_to_end_slice_encoder_forward(self):
        """Full forward pass: batch -> slice_encoder -> (B, D, embed_dim)."""
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            _make_synthetic_dataset(data_root, num_samples=2)

            _, val_tf = get_transforms(
                model_name=self.model_name,
                num_slices=self.num_slices,
                spatial_mode="resample",
            )
            ds = SliceDataset(
                path_root=str(data_root),
                split="train",
                transform=val_tf,
                plane="axial",
            )
            dl = torch.utils.data.DataLoader(
                ds,
                batch_size=2,
                shuffle=False,
                collate_fn=slice_collate_fn,
            )
            batch = next(iter(dl))
            x = batch["x"].cuda().float()

            encoder = build_slice_encoder(self.model_name, freeze=True)
            encoder = encoder.cuda().eval()
            with torch.no_grad():
                feats = encoder(x)

            embed_dim = self.cfg["embed_dim"]
            self.assertEqual(
                feats.shape,
                (2, self.num_slices, embed_dim),
                f"Expected (2, {self.num_slices}, {embed_dim})",
            )
            self.assertTrue(torch.isfinite(feats).all())


if __name__ == "__main__":
    unittest.main()
