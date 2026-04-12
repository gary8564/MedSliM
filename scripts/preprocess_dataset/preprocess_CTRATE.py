"""
Preprocess CT-RATE dataset for MedSliM.

Flattens the nested patient/scan/reconstruction NIfTI layout into a single
directory per split and merges multi-abnormality label CSVs.

Source layout:
  {data_dir}/{train,valid}_fixed/{patient_id}/{scan_id}/{volume}.nii.gz
  {data_dir}/multi_abnormality_labels/{train,valid}_predicted_labels.csv

Output layout:
  {save_dir}/{split}/{volume_name}.nii.gz + metadata.csv
"""
import argparse
import logging
import os
import shutil
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)

SPLITS = {
    "train_fixed": "train",
    "valid_fixed": "valid",
}


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)


def collect_nifti_tasks(data_dir: Path, save_dir: Path) -> list[dict]:
    """Walk all split directories and collect NIfTI files."""
    tasks = []
    for src_name, split in SPLITS.items():
        split_dir = data_dir / src_name
        if not split_dir.exists():
            logger.warning(f"Split directory not found: {split_dir}")
            continue

        for patient_dir in sorted(split_dir.iterdir()):
            if not patient_dir.is_dir():
                continue
            for scan_dir in sorted(patient_dir.iterdir()):
                if not scan_dir.is_dir():
                    continue
                for nifti in sorted(scan_dir.glob("*.nii.gz")):
                    tasks.append({
                        "source": str(nifti),
                        "volume_name": nifti.name,
                        "split": split,
                        "patient_id": patient_dir.name,
                        "scan_id": scan_dir.name,
                        "save_dir": str(save_dir),
                    })
    return tasks


def load_labels(data_dir: Path) -> pd.DataFrame:
    """Load and concatenate multi-abnormality label CSVs."""
    label_dir = data_dir / "multi_abnormality_labels"
    dfs = []
    for csv_name in ["train_predicted_labels.csv", "valid_predicted_labels.csv"]:
        csv_path = label_dir / csv_name
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(df)} entries from {csv_name}")
            dfs.append(df)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Preprocess CT-RATE: flatten NIfTI + merge labels")
    parser.add_argument(
        "--data-dir", type=str,
        default="/hpcwork/rwth1833/datasets/CT-RATE/dataset",
        help="Root folder containing CT-RATE dataset directory",
    )
    parser.add_argument(
        "--save-dir", type=str,
        default="/hpcwork/rwth1833/datasets/preprocessed/CT-RATE",
        help="Output directory for organized NIfTI files",
    )
    parser.add_argument(
        "--symlink", action="store_true",
        help="Create symlinks instead of copying files (saves disk space)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Step 1: Collecting NIfTI files ...")
    logger.info("=" * 60)
    tasks = collect_nifti_tasks(data_dir, save_dir)
    logger.info(f"Found {len(tasks)} NIfTI files")

    logger.info("=" * 60)
    logger.info("Step 2: Loading multi-abnormality labels ...")
    logger.info("=" * 60)
    df_labels = load_labels(data_dir)
    label_map = {}
    if not df_labels.empty and "VolumeName" in df_labels.columns:
        label_map = {row["VolumeName"]: row.to_dict() for _, row in df_labels.iterrows()}
        logger.info(f"Label entries loaded: {len(label_map)}")

    logger.info("=" * 60)
    logger.info("Step 3: Organizing files into split directories ...")
    logger.info("=" * 60)

    metadata_rows = []
    for task in tqdm(tasks, desc="Organizing"):
        split = task["split"]
        volume_name = task["volume_name"]
        out_dir = save_dir / split
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / volume_name

        if not out_path.exists():
            source = Path(task["source"])
            if args.symlink:
                os.symlink(source.resolve(), out_path)
            else:
                shutil.copy2(source, out_path)

        row = {
            "ID": volume_name.replace(".nii.gz", ""),
            "volume_name": volume_name,
            "split": split,
            "patient_id": task["patient_id"],
            "scan_id": task["scan_id"],
            "nifti_path": str(out_path),
        }

        if volume_name in label_map:
            label_row = label_map[volume_name]
            for col, val in label_row.items():
                if col != "VolumeName":
                    row[col] = val

        metadata_rows.append(row)

    df = pd.DataFrame(metadata_rows)
    df.to_csv(save_dir / "metadata.csv", index=False)

    logger.info("=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    for split in sorted(df["split"].unique()):
        count = len(df[df["split"] == split])
        logger.info(f"  {split}/: {count} files")
    matched = len([r for r in metadata_rows if len(r) > 6])
    logger.info(f"Label-matched: {matched}/{len(metadata_rows)}")
    logger.info("=" * 60)
    logger.info("Preprocessing completed.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
