import argparse
from pathlib import Path
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from safetensors.torch import save_file

from med_slim.model.slice_encoder import build_slice_encoder
from med_slim.data.slice_dataset import SliceDataset, slice_collate_fn
from med_slim.utils.preprocessing.transforms import get_transforms

def main():
    parser = argparse.ArgumentParser(description="Precompute slice features")
    parser.add_argument("--data-dir", type=str, default="/hpcwork/rwth1833/datasets/preprocessed/MRNet", help="Root folder of the dataset to precompute.")
    parser.add_argument("--save-dir", type=str, default="/hpcwork/rwth1833/feat_caches/MRNet", help="Output directory for precomputed features.")
    parser.add_argument("--plane", type=str, default="axial", choices=["axial", "sagittal", "coronal"], help="Plane of the slices to be precomputed.")
    parser.add_argument("--num-slices", type=int, default=32, help="Number of slices along depth used by transforms.")
    parser.add_argument("--amp", action="store_true", help="Use automatic mixed precision on CUDA")
    parser.add_argument("--model-name", type=str, default="dinov2", choices=["ark", "dinov2", "rad-dino", "medsiglip", "biomedclip"], help="Slice encoder backbone.")
    parser.add_argument("--model-repo", type=str, default=None, help="Optional HF repo override for DINO/MedSigLIP/CLIP.")
    parser.add_argument("--ark-checkpoint", type=str, default=None, help="Ark checkpoint path (required if --model-name ark).")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Dataset split to precompute.")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--batch-size", type=int, default=8, help="Number of studies per batch")
    args = parser.parse_args()

    out_dir = Path(args.save_dir) / f"slices_{args.num_slices}" / args.model_name / args.split / args.plane
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")

    # Build slice encoder (expects [B, C, W, H, D] with C=1 grayscale)
    if args.model_name == "ark":
        ckpt = args.ark_checkpoint
        if ckpt is None or not os.path.exists(ckpt):
            raise FileNotFoundError("Ark checkpoint not provided. Use --ark-checkpoint.")
        slice_encoder = build_slice_encoder(name="ark", checkpoint=ckpt, freeze=True).to(device).eval()
    else:
        slice_encoder = build_slice_encoder(name=args.model_name, model_repo=args.model_repo, freeze=True).to(device).eval()

    _, val_tf = get_transforms(model_name=args.model_name, num_slices=args.num_slices)

    ds = SliceDataset(
        path_root=args.data_dir,
        split=args.split,
        transform=val_tf,
        plane=args.plane,
    )
    
    data_loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=slice_collate_fn,
    )
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Precomputing slice features"):
            uids = batch["uid"]
            x = batch["x"].to(device=device, dtype=torch.float32, non_blocking=True)  # (B, C, W, H, D)
            with torch.autocast("cuda", enabled=args.amp):
                feats = slice_encoder(x)  # (B, D, embed_dim)
            feats = feats.cpu()

            # Save one file per case in the batch
            for i, uid in enumerate(uids):
                save_path = out_dir / f"{uid:04d}.safetensors"
                tensor_dict = {"feats": feats[i].to(dtype=torch.float16)}  
                metadata = {
                    "uid": str(int(uid)),
                    "plane": str(args.plane),
                    "model_name": str(args.model_name),
                    "num_slices": str(int(args.num_slices)),
                }
                save_file(tensor_dict, str(save_path), metadata=metadata)


if __name__ == "__main__":
    main()