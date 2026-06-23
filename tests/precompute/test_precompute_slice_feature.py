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
from safetensors.torch import load_file, save_file
from pathlib import Path

from med_slim.data.slice_dataset import (
    SliceDataset,
    TiledSliceDataset,
    build_pre_tile_transform,
    slice_collate_fn,
    tiled_slice_collate_fn,
)
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
                    plane="axial",
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
                plane="axial",
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
                plane="axial",
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
                plane="axial",
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

    def test_tiled_slice_dataset(self):
        """Tiled dataset should return [B, num_tiled_regions, C, W, H, num_slices]."""
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            _make_synthetic_dataset(data_root, num_samples=2)

            grid_size = 2
            pre_tile_tf = build_pre_tile_transform(plane="axial", crop_empty_slices=False)
            _, view_tf = get_transforms(
                model_name=self.model_name,
                plane="axial",
                num_slices=self.num_slices,
                spatial_mode="resize",
            )
            ds = TiledSliceDataset(
                path_root=str(data_root),
                split="train",
                pre_tile_transform=pre_tile_tf,
                view_transform=view_tf,
                grid_size=grid_size,
                plane="axial",
            )
            dl = torch.utils.data.DataLoader(
                ds,
                batch_size=2,
                shuffle=False,
                collate_fn=tiled_slice_collate_fn,
            )
            batch = next(iter(dl))
            x = batch["x"]
            num_regions = 1 + grid_size * grid_size
            self.assertEqual(x.shape, (2, num_regions, 1, self.W_crop, self.H_crop, self.num_slices))
            self.assertEqual(len(batch["region_boxes"][0]), num_regions)
            self.assertEqual(batch["region_boxes"][0][0], [0.0, 0.0, 1.0, 1.0])

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required for encoder test")
    def test_end_to_end_slice_encoder_forward(self):
        """Full forward pass: batch -> slice_encoder -> (B, D, embed_dim)."""
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            _make_synthetic_dataset(data_root, num_samples=2)

            _, val_tf = get_transforms(
                model_name=self.model_name,
                plane="axial",
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

    def test_tiled_feats_permute_saves_to_safetensors(self):
        batch_size, num_tiled_regions, num_slices, embed_dim = 2, 5, 12, 64
        view_feats = torch.randn(batch_size * num_tiled_regions, num_slices, embed_dim)
        feats = view_feats.reshape(
            batch_size,
            num_tiled_regions,
            num_slices,
            embed_dim,
        ).permute(0, 2, 1, 3).contiguous().cpu().to(dtype=torch.float32)

        with tempfile.TemporaryDirectory() as tmp:
            for i in range(batch_size):
                save_path = Path(tmp) / f"{i}.safetensors"
                save_file({"feats": feats[i].contiguous()}, str(save_path))
                loaded = load_file(str(save_path))
                self.assertEqual(
                    tuple(loaded["feats"].shape),
                    (num_slices, num_tiled_regions, embed_dim),
                )

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required for encoder test")
    def test_tiled_end_to_end_slice_encoder_forward_and_save(self):
        """Tiled batch -> encoder -> permute -> safetensors save (matches precompute path)."""
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp) / "data"
            out_dir = Path(tmp) / "out"
            out_dir.mkdir()
            _make_synthetic_dataset(data_root, num_samples=1)

            grid_size = 2
            regional_tokens = grid_size * grid_size
            num_regions = 1 + regional_tokens
            pre_tile_tf = build_pre_tile_transform(plane="axial", crop_empty_slices=False)
            _, view_tf = get_transforms(
                model_name=self.model_name,
                plane="axial",
                num_slices=self.num_slices,
                spatial_mode="crop",
            )
            ds = TiledSliceDataset(
                path_root=str(data_root),
                split="train",
                pre_tile_transform=pre_tile_tf,
                view_transform=view_tf,
                grid_size=grid_size,
                plane="axial",
            )
            batch = tiled_slice_collate_fn([ds[0]])
            x = batch["x"].cuda().float()

            encoder = build_slice_encoder(self.model_name, freeze=True).cuda().eval()
            with torch.no_grad():
                batch_size, num_tiled_regions, channels, width, height, num_slices = x.shape
                x_views = x.reshape(
                    batch_size * num_tiled_regions,
                    channels,
                    width,
                    height,
                    num_slices,
                )
                view_feats = encoder(x_views)
                feats = view_feats.reshape(
                    batch_size,
                    num_tiled_regions,
                    num_slices,
                    -1,
                ).permute(0, 2, 1, 3).contiguous().cpu().to(dtype=torch.float32)

            save_path = out_dir / f"{batch['uid'][0]}.safetensors"
            save_file({"feats": feats[0].contiguous()}, str(save_path), metadata={"regional_tokens": str(regional_tokens)})
            loaded = load_file(str(save_path))
            embed_dim = self.cfg["embed_dim"]
            self.assertEqual(
                tuple(loaded["feats"].shape),
                (self.num_slices, num_regions, embed_dim),
            )

    def test_subset_wrapped_dataloader(self):
        """Precompute keeps slice_ds for metadata while loader_ds is wrapped for the DataLoader."""
        with tempfile.TemporaryDirectory() as tmp:
            data_root = Path(tmp)
            _make_synthetic_dataset(data_root, num_samples=3)

            slice_ds = SliceDataset(
                path_root=str(data_root),
                split="train",
                plane="axial",
            )
            loader_ds = torch.utils.data.Subset(slice_ds, [0, 2])
            loader_ds = torch.utils.data.Subset(loader_ds, [0])

            uid = loader_ds[0]["uid"]
            self.assertEqual(
                slice_ds.get_nifti_path(uid),
                data_root / "train" / "axial" / f"{uid}.nii.gz",
            )


if __name__ == "__main__":
    unittest.main()
