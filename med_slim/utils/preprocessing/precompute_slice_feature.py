import argparse
import os
import glob
import gc
import torch
import numpy as np
import nibabel as nib
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from safetensors.torch import save_file
from pathlib import Path

from med_slim.model.slice_encoder import build_slice_encoder
from med_slim.data.slice_dataset import SliceDataset, slice_collate_fn
from med_slim.utils.preprocessing.transforms import get_transforms, get_adaptive_transform


# Preprocessing modes available for experimentation
# Each mode can be used as a form of augmentation in SSL
SPATIAL_MODES = ["resize", "resample", "crop", "adaptive"]

def get_num_slices(data_dir: str, split: str, plane: str) -> int:
    pattern = os.path.join(data_dir, split, plane, '*.nii.gz')
    nifti_file_paths = glob.glob(pattern)
    num_slices = []
    for nifti_file_path in nifti_file_paths:
        image = nib.load(nifti_file_path).get_fdata()
        num_slices.append(image.shape[2])
    return int(np.ceil(np.mean(num_slices)))

def build_transform(model_name: str, spatial_mode: str, num_slices: int = None):
    """
    Build the appropriate transform based on spatial mode.
    
    Args:
        model_name: Name of the pretrained model (e.g., 'ark', 'dinov2', 'dinov3', 'rad-dino', 'medsiglip', 'biomedclip')
        spatial_mode: Preprocessing strategy:
            - 'resize': Scale in-plane to target size (fast, may distort aspect ratio)
            - 'resample': Resample maintaining physical spacing (preserves anatomy)
            - 'crop': CropOrPad to target size (preserves native resolution)
            - 'adaptive': Smart selection based on source/target resolution ratio
        num_slices: Number of slices for depth dimension (None = keep original)
    
    Returns:
        val_transform: The validation/inference transform (no augmentation)
    """
    if spatial_mode == "adaptive":
        # AdaptivePreprocessing chooses optimal strategy based on resolution
        return get_adaptive_transform(
            model_name=model_name,
            num_slices=num_slices,
            to_tensor=False,
        )
    else:
        # Standard transforms with specified spatial mode
        _, val_transform = get_transforms(
            model_name=model_name,
            num_slices=num_slices,
            spatial_mode=spatial_mode,
            to_tensor=False,
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
                """
    )
    parser.add_argument("--data-dir", type=str, default="/hpcwork/rwth1833/datasets/preprocessed/MRNet", help="Root folder of the dataset to precompute.")
    parser.add_argument("--save-dir", type=str, default="/hpcwork/rwth1833/feat_caches/MRNet", help="Output directory for precomputed features.")
    parser.add_argument("--plane", type=str, default="axial", choices=["axial", "sagittal", "coronal"], help="Plane of the slices to be precomputed.")
    parser.add_argument("--spatial-mode", type=str, default="crop", choices=SPATIAL_MODES,
                        help="Preprocessing strategy: "
                             "'resize' (scale to target), 'resample' (maintain spacing), "
                             "'crop' (crop/pad to target), 'adaptive' (adaptive selection). "
                             "Default: crop")
    parser.add_argument("--use-raw-slice-resolution", action="store_true", help="Use raw slice resolution instead of cropped or padded resolution.")
    parser.add_argument("--num-slices", type=int, default=None, help="Number of slices along depth used by transforms.")
    parser.add_argument("--amp", type=str, default=None, choices=["fp16", "bf16"], 
                        help="Use automatic mixed precision: 'fp16' (faster but can overflow) or 'bf16' (safer, recommended)")
    parser.add_argument("--model-name", type=str, default="dinov2", choices=["ark", "dinov2", "dinov3", "rad-dino", "medsiglip", "biomedclip", "mri-core"], help="Slice encoder backbone.")
    parser.add_argument("--model-repo", type=str, default=None, help="Optional HF repo override for DINO/MedSigLIP/CLIP.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Local checkpoint path (required for --model-name ark or mri-core).")
    parser.add_argument("--local-cache-dir", type=str, default=None, help="Local cache directory to store the model.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Dataset split to precompute.")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--min-slices", type=int, default=10,
                        help="Minimum number of slices required. Volumes with fewer slices "
                             "(e.g., scout/localizer scans) are skipped. Default: 10")
    # Optimization options
    parser.add_argument("--shard-id", type=int, default=0, help="Shard ID for parallel processing (0-indexed)")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards for parallel processing")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for faster inference")
    args = parser.parse_args()
    device = torch.device("cuda")
    
    # Validate spatial mode
    spatial_mode = args.spatial_mode
    if spatial_mode not in SPATIAL_MODES:
        raise ValueError(f"Unknown spatial_mode: {spatial_mode}. Choose from {SPATIAL_MODES}")

    # Build slice encoder (expects [B, C, W, H, D] with C=1 grayscale)
    slice_encoder = build_slice_encoder(name=args.model_name, model_repo=args.model_repo, checkpoint=args.checkpoint, local_cache_dir=args.local_cache_dir, freeze=True).to(device).eval()
    
    # Optional: compile model for faster inference
    if args.compile:
        print("Compiling model with torch.compile()...")
        slice_encoder = torch.compile(slice_encoder)
    
    # Determine num_slices and build transforms
    if args.use_raw_slice_resolution:
        batch_size = 1
        num_slices = None
        num_slices_for_logging = "raw"
    else:
        batch_size = 4
        num_slices = args.num_slices if args.num_slices is not None else get_num_slices(args.data_dir, args.split, args.plane)
        num_slices_for_logging = str(num_slices)
    
    # Build transforms based on spatial mode
    image_transforms = build_transform(
        model_name=args.model_name,
        spatial_mode=spatial_mode,
        num_slices=num_slices,
    )
    
    print(f"Configuration:")
    print(f"  spatial_mode: {spatial_mode}")
    print(f"  num_slices: {num_slices_for_logging}")
    print(f"  batch_size: {batch_size}")
    print(f"  plane: {args.plane}")
    print(f"  split: {args.split}")
    print(f"  model_name: {args.model_name}")
    
    # Build output directory path: {save_dir}/slices_{num_slices}/{spatial_mode}/{model_name}/{split}/{plane}/
    # Including spatial_mode in path allows storing features from different preprocessing methods
    # This enables using different preprocessing as positive pairs in SSL
    out_dir = Path(args.save_dir) / f"slices_{num_slices_for_logging}" / spatial_mode / args.model_name / args.split / args.plane
    # out_dir = Path(args.save_dir) / f"slices_{num_slices_for_logging}" / args.model_name / args.split / args.plane
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = SliceDataset(
        path_root=args.data_dir,
        split=args.split,
        transform=image_transforms,
        plane=args.plane,
    )
    
    # Filter out volumes with too few slices (e.g., scout/localizer scans)
    if args.min_slices > 0:
        valid_indices = []
        skipped = []
        for idx in range(len(ds)):
            uid = ds.sample_ids[idx]
            img_path = ds.path_root / ds.split / ds.plane / f"{uid}.nii.gz"
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
        # Get indices for this shard: [shard_id, shard_id + num_shards, shard_id + 2*num_shards, ...]
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
        collate_fn=slice_collate_fn,
    )
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(data_loader, desc="Precomputing slice features")):
            uids = batch["uid"]
            x = batch["x"].to(device=device, dtype=torch.float32, non_blocking=True)  # (B, C, W, H, D)
            
            # Check for NaN in input
            if torch.isnan(x).any():
                raise ValueError(f"NaN in input for batch {batch_idx}, uids: {uids}")
            
            amp_enabled = args.amp is not None
            # bf16 is recommended (same exponent range as fp32, no overflow)
            amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16 
            with torch.autocast("cuda", enabled=amp_enabled, dtype=amp_dtype):
                feats = slice_encoder(x)  # (B, D, embed_dim)
            feats = feats.cpu().to(dtype=torch.float32)
            
            # Check for NaN/Inf in output
            if torch.isnan(feats).any() or torch.isinf(feats).any():
                raise ValueError(f"NaN/Inf in features for batch {batch_idx}, uids: {uids}")

            # Save one file per case in the batch
            for i, uid in enumerate(uids):
                save_path = out_dir / f"{uid}.safetensors"
                tensor_dict = {"feats": feats[i]}  
                metadata = {
                    "uid": str(uid),
                    "plane": str(args.plane),
                    "model_name": str(args.model_name),
                    "num_slices": num_slices_for_logging,
                    "spatial_mode": spatial_mode,
                }
                save_file(tensor_dict, str(save_path), metadata=metadata)


if __name__ == "__main__":
    main()