import argparse
import logging
import sys
import pickle
from pathlib import Path
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torchio as tio
from tqdm import tqdm


logger = logging.getLogger(__name__)


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)


def pickle_to_nifti(task: dict) -> dict | None:
    """Convert a pickle volumetric file to NIfTI format."""
    try:
        vol_path = Path(task["vol_path"])
        save_dir = Path(task["save_dir"])
        split = task["split"]
        plane = task["plane"]
        # Use filename stem as UID (e.g., "502889-5" from "502889-5.pck")
        uid = vol_path.stem

        # Load pickle data
        with open(vol_path, "rb") as f:
            vol_data = pickle.load(f)

        # Data shape is (D, H, W), need to convert to (W, H, D) for torchio
        # torchio expects (C, W, H, D) where C is channel
        vol_data = np.swapaxes(vol_data, 0, -1)  # (D, H, W) -> (W, H, D)

        # Create NIfTI image
        img = tio.ScalarImage(tensor=vol_data[None])  # Add channel dimension

        # Save to output path
        out_dir = save_dir / split / plane
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{uid}.nii.gz"
        img.save(out_file)

        result = {
            "ID": uid,
            "nifti_path": str(out_file),
            "size": list(img.spatial_shape),
        }
        for key in task.get("metadata_cols", []):
            if key in task:
                result[key] = task[key]
        return result
    except Exception as e:
        logger.warning(f"Failed conversion for {task.get('vol_path', 'unknown')}: {e}")
        return None

def get_split_metadata(df_meta: pd.DataFrame, data_dir: Path, split: str) -> pd.DataFrame:
    """Filter metadata CSV based on files that exist in vol_{split} directory.
    
    Args:
        df_meta: metadata DataFrame
        data_dir: root directory containing vol_train and vol_test folders
        split: either 'train' or 'test'
    
    Returns:
        Filtered DataFrame containing only rows where volumeFilename exists in vol_{split}
    """
    vol_dir = data_dir / f"vol_{split}"
    if not vol_dir.exists():
        raise FileNotFoundError(f"Directory {vol_dir} does not exist")
    
    # Get all pickle filenames in the vol_{split} directory
    existing_files = {f.name for f in vol_dir.glob("*.pck")}
    logger.info(f"Found {len(existing_files)} files in {vol_dir}")
    
    # Filter metadata to only include rows where volumeFilename exists
    df_filtered = df_meta[df_meta["volumeFilename"].isin(existing_files)].copy()
    logger.info(f"Matched {len(df_filtered)} entries from metadata for {split} split")
    
    return df_filtered


def main():
    parser = argparse.ArgumentParser(description="Preprocess kneeMRI dataset to NIfTI")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Root folder containing kneeMRI dataset",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="Output directory for NIfTI and metadata",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split name used in output folder structure (e.g., train/val/test)",
    )
    parser.add_argument(
        "--plane",
        type=str,
        default="sagittal",
        help="Plane of the MRI scans (sagittal/coronal/axial)",
    )
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)

    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    logger.info("================================================")
    logger.info("Step 1: Loading metadata")
    logger.info("================================================")
    metadata_csv = data_dir / "metadata.csv"
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found at {metadata_csv}")

    # Read metadata CSV and filter to existing files
    df_meta = pd.read_csv(metadata_csv)
    logger.info(f"Loaded metadata with {len(df_meta)} entries")
    df_meta = get_split_metadata(df_meta, data_dir, args.split)
    metadata_cols = list(df_meta.columns)

    vol_dir = data_dir / f"vol_{args.split}"

    # Build tasks
    logger.info("================================================")
    logger.info("Step 2: Converting pickle to NIfTI")
    logger.info("================================================")
    tasks = []
    for _, row in df_meta.iterrows():
        vol_filename = row["volumeFilename"]
        vol_path = vol_dir / vol_filename

        task = {
            "vol_path": str(vol_path),
            "save_dir": str(save_dir),
            "split": args.split,
            "plane": args.plane,
            "metadata_cols": metadata_cols,
        }
        for col in metadata_cols:
            task[col] = row[col]
        tasks.append(task)

    logger.info(f"Found {len(tasks)} volumes to convert")

    # Parallel processing
    metadata_rows = []
    with Pool(processes=max(1, int(args.workers))) as pool:
        chunksize = max(1, len(tasks) // (max(1, int(args.workers)) * 4))
        for md in tqdm(pool.imap_unordered(pickle_to_nifti, tasks, chunksize=chunksize), total=len(tasks)):
            if md is not None:
                metadata_rows.append(md)

    # Save metadata CSV
    logger.info("================================================")
    logger.info("Step 3: Saving metadata CSV")
    logger.info("================================================")
    df_output = pd.DataFrame(metadata_rows)
    
    col_order = ["ID"] + [c for c in metadata_cols if c in df_output.columns]
    col_order += [c for c in df_output.columns if c not in col_order]
    df_output = df_output[col_order]
    
    # Produce two CSV variants: multiclass (injury severity) and binary (injury vs normal)
    if "aclDiagnosis" in df_output.columns:
        multiclass_counts = df_output["aclDiagnosis"].value_counts().sort_index()

        # Multiclass CSV: keep original 0/1/2 labels, just rename "aclDiagnosis" to "acl"
        df_multiclass = df_output.copy()
        df_multiclass.rename(columns={"aclDiagnosis": "acl"}, inplace=True)
        df_multiclass.to_csv(save_dir / f"{args.split}_multiclass.csv", index=False)
        logger.info(
            f"{args.split}_multiclass.csv written with {len(df_multiclass)} entries. "
            f"acl label distribution (0=normal, 1=partial, 2=complete): {multiclass_counts.to_dict()}"
        )

        # Binary CSV: convert labels for injury severity to binary labels
        df_binary = df_output.copy()
        df_binary["acl"] = (df_binary["aclDiagnosis"] > 0).astype(int)
        df_binary.drop(columns=["aclDiagnosis"], inplace=True)
        binary_counts = df_binary["acl"].value_counts().sort_index()
        df_binary.to_csv(save_dir / f"{args.split}_binary.csv", index=False)
        logger.info(
            f"{args.split}_binary.csv written with {len(df_binary)} entries. "
            f"acl label distribution (0=normal, 1=injury): {binary_counts.to_dict()}"
        )
    else:
        raise ValueError("aclDiagnosis column not found in metadata")

    num_files = len(list(save_dir.rglob("*.nii.gz")))
    logger.info(f"Finished. NIfTI files written: {num_files}")
    logger.info("================================================")
    logger.info("Preprocessing completed.")
    logger.info("================================================")


if __name__ == "__main__":
    main()
