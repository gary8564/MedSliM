import argparse
import json
import math
import os
import glob
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from safetensors.torch import save_file
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

from med_slim.model.slice_encoder import build_slice_encoder
from med_slim.data.slice_dataset import (
    SliceDataset,
    TiledSliceDataset,
    build_pre_tile_transform,
    slice_collate_fn,
    tiled_slice_collate_fn,
)
from med_slim.utils.preprocessing.transforms import get_transforms, get_adaptive_transform
from med_slim.utils.model_config import get_slice_encoder_config

load_dotenv()
SPATIAL_MODES = ["resize", "resample", "crop", "adaptive"]


def _parse_mri_sequences(raw: Optional[list[str]] = None) -> Optional[str | list[str]]:
    """Parse `--mri-sequences` CLI values into `SliceDataset` kwargs."""
    if raw is None:
        return None
    if len(raw) == 1 and raw[0].lower() == "none":
        return None
    if len(raw) == 1 and raw[0].lower() == "all":
        return "all"
    return raw


def get_num_slices(data_dir: str, split: str, plane: str) -> int:
    pattern = os.path.join(data_dir, split, plane, '*.nii.gz')
    nifti_file_paths = glob.glob(pattern)
    num_slices = []
    for nifti_file_path in nifti_file_paths:
        image = nib.load(nifti_file_path).get_fdata()
        num_slices.append(image.shape[2])
    return int(np.ceil(np.mean(num_slices)))


def build_transform(model_name: str, spatial_mode: str, plane: str,
                    num_slices: int = None, crop_empty_slices: bool = False):
    """Build the standard FM preprocessing transform."""
    if spatial_mode == "adaptive":
        return get_adaptive_transform(
            model_name=model_name,
            num_slices=num_slices,
            crop_empty_slices=crop_empty_slices,
            to_tensor=False,
            plane=plane,
        )
    else:
        _, val_transform = get_transforms(
            model_name=model_name,
            num_slices=num_slices,
            spatial_mode=spatial_mode,
            crop_empty_slices=crop_empty_slices,
            to_tensor=False,
            plane=plane,
        )
        return val_transform


def main():
    parser = argparse.ArgumentParser(
        description="Precompute slice features with different preprocessing strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
                Examples:
                # Precompute with resize
                python precompute_slice_feature.py --spatial-mode resize
                
                # Precompute with resample
                python precompute_slice_feature.py --spatial-mode resample
                
                # Precompute with crop
                python precompute_slice_feature.py --spatial-mode crop
                
                # Precompute with adaptive selection based on source/target resolution ratio
                python precompute_slice_feature.py --spatial-mode adaptive
                
                # Precompute with adaptive selection based on source/target resolution ratio and trim near-empty edge slices
                python precompute_slice_feature.py --spatial-mode adaptive --crop-empty-slices
                """  
    )
    parser.add_argument("--data-dir", type=str, default="/hpcwork/rwth1833/datasets/preprocessed/MRNet",
                        help="Root folder of the dataset to precompute.")
    parser.add_argument("--save-dir", type=str, default="/hpcwork/rwth1833/feat_caches/MRNet",
                        help="Output directory for precomputed features.")
    parser.add_argument("--plane", type=str, default="axial",
                        choices=["axial", "sagittal", "coronal"],
                        help="Plane of the slices to be precomputed.")
    parser.add_argument("--spatial-mode", type=str, default="adaptive", choices=SPATIAL_MODES,
                        help="Preprocessing strategy (default: adaptive)")
    parser.add_argument("--use-raw-slice-resolution", action="store_true",
                        help="Use raw slice resolution instead of cropped or padded resolution.")
    parser.add_argument("--num-slices", type=int, default=None,
                        help="Number of slices along depth used by transforms.")
    parser.add_argument("--crop-empty-slices", action="store_true",
                        help="Trim near-empty edge slices before spatial transforms (recommended for "
                             "atlas-registered data like BraTS or datasets with padded volumes).")
    parser.add_argument("--amp", type=str, default=None, choices=["fp16", "bf16"],
                        help="Use automatic mixed precision: 'fp16' or 'bf16' (recommended)")
    parser.add_argument("--model-name", type=str, default="dinov2",
                        choices=["ark", "dinov2", "dinov3", "rad-dino", "medsiglip",
                                 "biomedclip", "mri-core", "medimageinsight"],
                        help="Slice encoder backbone.")
    parser.add_argument("--model-repo", type=str, default=None,
                        help="Optional HF repo override for DINO/MedSigLIP/CLIP.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Local checkpoint path (required for ark, mri-core, medimageinsight).")
    parser.add_argument("--local-cache-dir", type=str, default=None,
                        help="Local cache directory to store the model.")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val", "test"], help="Dataset split to precompute.")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--min-slices", type=int, default=10,
                        help="Minimum number of slices required. Volumes with fewer slices "
                             "(e.g., scout/localizer scans) are skipped. Default: 10")
    parser.add_argument("--shard-id", type=int, default=0,
                        help="Shard ID for parallel processing (0-indexed)")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of shards for parallel processing")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile for faster inference")
    parser.add_argument("--curia-token-mode", type=str, default="cls",
                        choices=["cls", "patch", "cls_patch"],
                        help="Curia only: token type to cache. "
                             "'cls' stores one embedding per slice; "
                             "'patch' stores flattened spatial patch tokens; "
                             "'cls_patch' stores both.")
    parser.add_argument("--curia-spatial-pool-kernel-size", type=int, default=None,
                        help="Curia only: optional average-pooling kernel over the "
                             "2D patch grid before flattening patch tokens.")
    parser.add_argument("--regional-tokens", type=int, default=0,
                        help="Tiled multi-crop CLS mode. 0 (default) stores one global "
                             "token per slice [num_slices, embed_dim]. N specifies the total number of "
                             "regional crop tokens per slice excluding the global token, "
                             "producing [num_slices, 1+N, embed_dim]. "
                             "Example: --regional-tokens 4 -> global + 2x2 tiles (5 tokens/slice).")
    parser.add_argument(
        "--mri-sequences",
        nargs="*",
        default=None,
        help="Define the MRI sequence types when the dataset contains multiple MRI sequence subfolders under {split}/ (e.g. pd pd_fs t2 t2_fs). "
             "Use 'all' to auto-discover sequence folders. "
             "If datasets do not contain multi-sequence, set to None for default {split}/{plane}/ layout (default).",
    )
    args = parser.parse_args()
    device = torch.device("cuda")

    spatial_mode = args.spatial_mode
    mri_sequences = _parse_mri_sequences(args.mri_sequences)
    if spatial_mode not in SPATIAL_MODES:
        raise ValueError(f"Unknown spatial_mode: {spatial_mode}. Choose from {SPATIAL_MODES}")

    # Validate tiled multi-crop configuration
    regional_tokens = args.regional_tokens
    if regional_tokens < 0:
        raise ValueError(f"--regional-tokens must be >= 0, got {regional_tokens}.")
    tiled_mode = regional_tokens > 0
    grid_size = 0
    if tiled_mode:
        grid_size = math.isqrt(regional_tokens)
        if grid_size * grid_size != regional_tokens:
            raise ValueError(
                f"--regional-tokens must be a perfect square (e.g. 4 for 2x2, 9 for 3x3), "
                f"got {regional_tokens}."
            )

    # Build slice encoder
    slice_encoder = build_slice_encoder(
        name=args.model_name, model_repo=args.model_repo,
        checkpoint=args.checkpoint, local_cache_dir=args.local_cache_dir,
        freeze=True,
        curia_token_mode=args.curia_token_mode,
        curia_spatial_pool_kernel_size=args.curia_spatial_pool_kernel_size,
    ).to(device).eval()

    if args.compile:
        print("Compiling model with torch.compile()...")
        slice_encoder = torch.compile(slice_encoder)

    # Determine num_slices and batch size
    fm_h = fm_w = None
    if tiled_mode:
        # Tiled multi-crop CLS creates global and regional views before the standard FM preprocessing transform.
        encoder_cfg = get_slice_encoder_config(args.model_name)
        fm_h, fm_w = tuple(encoder_cfg["img_size"])
    if args.use_raw_slice_resolution:
        batch_size = 1
        num_slices = None
        num_slices_for_logging = "raw"
    else:
        batch_size = 4
        num_slices = (args.num_slices if args.num_slices is not None 
                      else get_num_slices(args.data_dir, args.split, args.plane))
        num_slices_for_logging = str(num_slices)

    if tiled_mode:
        pre_tile_transform = build_pre_tile_transform(
            plane=args.plane,
            crop_empty_slices=args.crop_empty_slices,
        )
        view_transform = build_transform(
            model_name=args.model_name,
            spatial_mode=spatial_mode,
            num_slices=num_slices,
            crop_empty_slices=False,
            plane=args.plane,
        )
    else:
        image_transforms = build_transform(
            model_name=args.model_name,
            spatial_mode=spatial_mode,
            num_slices=num_slices,
            crop_empty_slices=args.crop_empty_slices,
            plane=args.plane,
        )

    # Region layout metadata when tiled_mode is enabled.
    num_regions = 1 + regional_tokens if tiled_mode else 1
    region_order = None
    if tiled_mode:
        labels = ["global"]
        for r in range(grid_size):
            for c in range(grid_size):
                labels.append(f"r{r}c{c}")
        region_order = ",".join(labels)

    print("Configuration:")
    print(f"  spatial_mode: {spatial_mode}")
    print(f"  num_slices: {num_slices_for_logging}")
    print(f"  crop_empty_slices: {args.crop_empty_slices}")
    print(f"  batch_size: {batch_size}")
    print(f"  plane: {args.plane}")
    print(f"  split: {args.split}")
    print(f"  model_name: {args.model_name}")
    if mri_sequences is not None:
        print(f"  mri_sequences: {mri_sequences}")
    if tiled_mode:
        print(f"  regional_tokens: {regional_tokens} (tile grid {grid_size}x{grid_size}, "
              f"{num_regions} tokens/slice)")
        print(f"  tiling: crop at original resolution, resize each crop to FM size (H, W)={(fm_h, fm_w)}")
    if args.model_name == "curia":
        print(f"  curia_token_mode: {args.curia_token_mode}")
        print(f"  curia_spatial_pool_kernel_size: {args.curia_spatial_pool_kernel_size}")

    folder_name = spatial_mode
    if tiled_mode:
        folder_name = f"{spatial_mode}_tiled_{grid_size}x{grid_size}"
    out_dir = (Path(args.save_dir) / f"slices_{num_slices_for_logging}"
               / folder_name / args.model_name / args.split / args.plane)
    out_dir.mkdir(parents=True, exist_ok=True)

    if tiled_mode:
        ds = TiledSliceDataset(
            path_root=args.data_dir,
            split=args.split,
            pre_tile_transform=pre_tile_transform,
            view_transform=view_transform,
            grid_size=grid_size,
            plane=args.plane,
            mri_sequences=mri_sequences,
        )
    else:
        ds = SliceDataset(
            path_root=args.data_dir,
            split=args.split,
            transform=image_transforms,
            plane=args.plane,
            mri_sequences=mri_sequences,
        )

    # Filter out volumes with too few slices
    if args.min_slices > 0:
        valid_indices = []
        skipped = []
        for idx in range(len(ds)):
            uid = ds.sample_ids[idx]
            img_path = ds.get_nifti_path(uid)
            n_slices = nib.load(str(img_path)).shape[2]
            if n_slices >= args.min_slices:
                valid_indices.append(idx)
            else:
                skipped.append((uid, n_slices))
        if skipped:
            print(f"Skipping {len(skipped)} volumes with less than {args.min_slices} slices:")
            for uid, n in skipped:
                print(f"  {uid}: {n} slices")
            ds = Subset(ds, valid_indices)

    # Apply sharding for parallel processing
    if args.num_shards > 1:
        all_indices = list(range(len(ds)))
        shard_indices = [idx for idx in all_indices if idx % args.num_shards == args.shard_id]
        ds = Subset(ds, shard_indices)
        print(f"Shard {args.shard_id}/{args.num_shards}: Processing {len(ds)}/{len(all_indices)} samples")

    data_loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=tiled_slice_collate_fn if tiled_mode else slice_collate_fn,
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Precomputing slice features")):
            uids = batch["uid"]
            x = batch["x"].to(dtype=torch.float32)

            if torch.isnan(x).any():
                raise ValueError(f"NaN in input for batch {batch_idx}, uids: {uids}")

            amp_enabled = args.amp is not None
            amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16

            if tiled_mode:
                # Encode all resized views in one FM batch:
                # [batch_size, num_tiled_regions, C, W, H, num_slices] -> [batch_size*num_tiled_regions, C, W, H, num_slices].
                batch_size, num_tiled_regions, channels, width, height, num_slices = x.shape
                x_views = x.reshape(
                    batch_size * num_tiled_regions,
                    channels,
                    width,
                    height,
                    num_slices,
                ).to(device=device, dtype=torch.float32, non_blocking=True)
                with torch.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
                    view_feats = slice_encoder(x_views)  # [batch_size*num_tiled_regions, num_slices, embed_dim]
                feats = view_feats.reshape(
                    batch_size,
                    num_tiled_regions,
                    num_slices,
                    -1,
                ).permute(0, 2, 1, 3).contiguous().cpu().to(dtype=torch.float32)
            else:
                x = x.to(device=device, non_blocking=True)
                with torch.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
                    feats = slice_encoder(x)  # (B, D, embed_dim)
                feats = feats.cpu().to(dtype=torch.float32)

            if torch.isnan(feats).any() or torch.isinf(feats).any():
                raise ValueError(f"NaN/Inf in features for batch {batch_idx}, uids: {uids}")

            for i, uid in enumerate(uids):
                save_path = out_dir / f"{uid}.safetensors"
                tensor_dict = {"feats": feats[i].contiguous()}
                metadata = {
                    "uid": str(uid),
                    "plane": str(args.plane),
                    "model_name": str(args.model_name),
                    "num_slices": num_slices_for_logging,
                    "spatial_mode": spatial_mode,
                }
                if tiled_mode:
                    metadata["regional_tokens"] = str(regional_tokens)
                    metadata["tile_grid"] = f"{grid_size}x{grid_size}"
                    metadata["num_regions"] = str(num_regions)
                    metadata["region_order"] = region_order
                    metadata["include_global_cls"] = "true"
                    metadata["region_boxes"] = json.dumps(batch["region_boxes"][i])
                if args.model_name == "curia":
                    metadata["curia_token_mode"] = str(args.curia_token_mode)
                    metadata["curia_spatial_pool_kernel_size"] = str(
                        args.curia_spatial_pool_kernel_size
                    )
                # Extract inter-slice spacing from the NIfTI header
                nifti_path = ds.get_nifti_path(uid)
                if nifti_path.is_file():
                    zooms = nib.load(str(nifti_path)).header.get_zooms()
                    # Depth axis (axis 2 in (C,W,H,D) convention) gives slice spacing
                    metadata["slice_spacing_mm"] = str(float(zooms[2]))
                save_file(tensor_dict, str(save_path), metadata=metadata)


if __name__ == "__main__":
    main()
