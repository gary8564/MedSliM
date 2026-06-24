"""
Patch existing safetensors feature caches with ``slice_spacing_mm`` metadata.

Reads each safetensors file, looks up the matching NIfTI header to extract
inter-slice spacing (zooms[2]), and rewrites the file in-place with the
original feats tensor and updated metadata.  No GPU required.

Usage:
    python patch_slice_spacing_metadata.py \
        --feat-dir /hpcwork/rwth1833/feat_caches/MRNet/slices_raw/crop \
        --data-dir /hpcwork/rwth1833/datasets/preprocessed/MRNet \
        --split train

    # Parallel across datasets:
    python patch_slice_spacing_metadata.py \
        --feat-dir /hpcwork/rwth1833/feat_caches/fastMRI/slices_raw/adaptive \
        --data-dir /hpcwork/rwth1833/datasets/preprocessed/fastMRI \
        --split train --workers 16
"""

import argparse
import os
from glob import glob
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import nibabel as nib
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm


def patch_one_file(
    safetensor_path: str,
    data_dir: str,
    split: str,
    dry_run: bool = False,
) -> str | None:
    """
    Patch a single safetensors file with slice_spacing_mm.

    Uses the safetensors filename (not the metadata uid) to locate the
    matching NIfTI at ``{data_dir}/{split}/{plane}/{uid}.nii.gz``.

    Returns:
        Status string for logging, or None if skipped.
    """
    with safe_open(safetensor_path, framework="pt", device="cpu") as f:
        metadata = f.metadata()
        feats = f.get_tensor("feats")

    if "slice_spacing_mm" in metadata:
        return None  # already patched

    plane = metadata["plane"]
    file_uid = os.path.splitext(os.path.basename(safetensor_path))[0]
    nifti_path = Path(data_dir) / split / plane / f"{file_uid}.nii.gz"
    if not nifti_path.is_file():
        return f"MISSING_NIFTI: {file_uid} (plane={plane})"

    zooms = nib.load(str(nifti_path)).header.get_zooms()
    slice_spacing_mm = float(zooms[2])

    metadata["slice_spacing_mm"] = str(slice_spacing_mm)

    if not dry_run:
        save_file({"feats": feats}, safetensor_path, metadata=metadata)

    return f"OK: {file_uid} plane={plane} spacing={slice_spacing_mm:.4f}mm"


def main():
    parser = argparse.ArgumentParser(
        description="Patch safetensors metadata with slice_spacing_mm from NIfTI headers."
    )
    parser.add_argument(
        "--feat-dir", type=str, required=True,
        help="Root of the feature cache (e.g. .../feat_caches/MRNet/slices_raw/crop)",
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Root of the preprocessed dataset with NIfTI files (e.g. .../datasets/preprocessed/MRNet)",
    )
    parser.add_argument(
        "--split", type=str, default="train", choices=["train", "val", "test"],
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Number of parallel I/O threads (default: 8)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Read NIfTI headers but don't write safetensors (for verification)",
    )
    args = parser.parse_args()

    # Discover all safetensors: {feat_dir}/{model}/train/{plane}/*.safetensors
    pattern = os.path.join(args.feat_dir, "*", args.split, "*", "*.safetensors")
    all_files = sorted(glob(pattern))

    if not all_files:
        print(f"No safetensors files found matching: {pattern}")
        raise SystemExit(1)

    print(f"Found {len(all_files)} safetensors files to patch")
    print(f"NIfTI source: {args.data_dir}")
    print(f"Split: {args.split}")
    if args.dry_run:
        print("DRY RUN — no files will be modified")

    patched = 0
    skipped = 0
    errors = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(patch_one_file, path, args.data_dir, args.split, args.dry_run): path
            for path in all_files
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Patching"):
            path = futures[future]
            try:
                result = future.result()
                if result is None:
                    skipped += 1
                elif result.startswith("OK"):
                    patched += 1
                else:
                    errors.append(result)
            except Exception as e:
                errors.append(f"ERROR on {path}: {e}")

    print(f"\nDone: {patched} patched, {skipped} already had spacing, {len(errors)} errors")
    if errors:
        print("Errors:")
        for err in errors[:20]:
            print(f"  {err}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")


if __name__ == "__main__":
    main()
