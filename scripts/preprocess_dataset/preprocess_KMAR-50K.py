"""
Preprocess KMAR-50K dataset for MedSliM SSL pretraining.

This script:
1. Collects all GroundTruthData NIfTI files (excluding _N4_Norm_RegSyN versions)
2. Parses filenames to extract study_id and view plane
3. Organizes files into {save_dir}/train/{plane}/ structure
4. Merges TrainingCohort.csv and TestingCohort.csv into a single train.csv

Filename format: {year}_{id}_{plane}_{version}.nii.gz
- year: 2020, 2021, 2022, 2023
- id: numeric patient/study ID (or date-based ID like 20210630003548)
- plane: sagittal, coronal, transection (→ axial)
- version: 0.0, 1.0, etc. (multiple series per study)

Output filename: {year}_{id}_MR{version_int}.nii.gz
    e.g. 2020_104_MR0.nii.gz, 2021_130_MR0.nii.gz
"""
import argparse
import logging
import re
import shutil
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Plane mapping: transection → axial
PLANE_MAP = {
    "sagittal": "sagittal",
    "coronal": "coronal",
    "transection": "axial",
    "axial": "axial",
}


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


def parse_filename(filename: str) -> dict | None:
    """
    Parse KMAR-50K filename to extract study_id, plane, and version.
    Common structure: {year}_{patient_id}_{plane}_{version}.nii.gz
    """
    # Skip processed versions (N4 bias corrected, normalized, registered)
    if "_N4_Norm_RegSyN" in filename:
        return None

    stem = filename.replace(".nii.gz", "")

    # - year: 4 digits
    # - patient_id: anything until we hit the plane keyword
    # - plane: sagittal, coronal, or transection
    # - version: digits.digits
    pattern = r"^(\d{4})_(.+?)_(sagittal|coronal|transection)_(\d+\.\d+)$"
    match = re.match(pattern, stem, re.IGNORECASE)

    if not match:
        logger.warning(f"Could not parse filename: {filename}")
        return None

    year, patient_id, plane_raw, version = match.groups()
    plane = PLANE_MAP.get(plane_raw.lower())

    if plane is None:
        raise ValueError(f"Unknown plane '{plane_raw}' in filename: {filename}")

    # Extract just the numeric prefix of patient_id for a clean study_id
    # e.g. "144-20210515001368" → "144", "383XX20221031001450" → "383"
    id_match = re.match(r"^(\d+)", patient_id)
    short_id = id_match.group(1) if id_match else patient_id

    study_id = f"{year}_{short_id}"
    version_int = version.split(".")[0]
    output_filename = f"{study_id}_MR{version_int}.nii.gz"
    csv_key = f"{year}_{patient_id}_{plane_raw}_{version_int}.nii.gz"

    return {
        "study_id": study_id,
        "plane": plane,
        "plane_raw": plane_raw,
        "version": version,
        "patient_id_full": patient_id,
        "output_filename": output_filename,
        "original_filename": filename,
        "csv_key": csv_key,
    }


def collect_nifti_files(data_dirs: list[Path]) -> list[dict]:
    """Collect all GroundTruthData NIfTI files from specified directories."""
    files = []
    for data_dir in data_dirs:
        if not data_dir.exists():
            logger.warning(f"Directory not found: {data_dir}")
            continue
        for nifti_file in data_dir.glob("*.nii.gz"):
            parsed = parse_filename(nifti_file.name)
            if parsed:
                parsed["source_path"] = str(nifti_file)
                files.append(parsed)
    logger.info(f"Collected {len(files)} NIfTI files")
    return files


def load_and_merge_csvs(train_csv: Path, test_csv: Path) -> pd.DataFrame:
    """
    Load and merge TrainingCohort.csv and TestingCohort.csv into a unified
    DataFrame with a common csv_key for matching with NIfTI files.
    """
    dfs = []

    if train_csv.exists():
        df_train = pd.read_csv(train_csv)
        df_train["csv_key"] = df_train["NII_FileName"].apply(
            lambda x: str(x).strip().strip('"').strip("'") if pd.notna(x) else ""
        )
        df_train["original_split"] = "train"
        dfs.append(df_train)
        logger.info(f"Loaded {len(df_train)} entries from TrainingCohort.csv")

    if test_csv.exists():
        df_test = pd.read_csv(test_csv)
        # TestingCohort uses a "Patient" column in format "2023_34_coronal_0"
        df_test["csv_key"] = df_test["Patient"].apply(
            lambda x: str(x).strip().strip('"').strip("'") + ".nii.gz" if pd.notna(x) else ""
        )
        df_test = df_test.rename(columns={"Patient": "NII_FileName"})
        df_test["original_split"] = "test"
        dfs.append(df_test)
        logger.info(f"Loaded {len(df_test)} entries from TestingCohort.csv")

    if not dfs:
        logger.warning("No CSV files found!")
        return pd.DataFrame()

    df_merged = pd.concat(dfs, ignore_index=True)
    logger.info(f"Merged CSV has {len(df_merged)} total entries")
    return df_merged


def main():
    parser = argparse.ArgumentParser(description="Preprocess KMAR-50K dataset for SSL pretraining")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Root folder containing KMAR-50K dataset")
    parser.add_argument("--save-dir", type=str, required=True,
                        help="Output directory for organized NIfTI files")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)

    data_root = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    ground_truth_dirs = [
        data_root / "KMAR-50K" / "GroundTruthData_part1",
        data_root / "KMAR-50K-part2" / "GroundTruthData_part2",
        data_root / "KMAR-50K" / "Testing _GroundTruthData" / "1_GroundTruthData",
    ]
    train_csv = data_root / "KMAR-50K" / "TrainingCohort.csv"
    test_csv = data_root / "KMAR-50K" / "TestingCohort.csv"

    logger.info("Step 1: Collecting GroundTruthData NIfTI files")
    files = collect_nifti_files(ground_truth_dirs)
    if not files:
        logger.error("No NIfTI files found!")
        return

    logger.info("Step 2: Loading and merging CSV metadata")
    df_csv = load_and_merge_csvs(train_csv, test_csv)
    csv_metadata = {}
    if not df_csv.empty:
        for _, row in df_csv.iterrows():
            key = row.get("csv_key", "")
            if key:
                csv_metadata[key] = row.to_dict()

    logger.info("Step 3: Organizing files into train/{plane}/ structure")
    metadata_rows = []
    planes = set()
    matched_count = 0
    unmatched_count = 0

    for file_info in tqdm(files, desc="Organizing files"):
        plane = file_info["plane"]
        output_filename = file_info["output_filename"]
        source_path = Path(file_info["source_path"])
        csv_key = file_info["csv_key"]

        out_dir = save_dir / "train" / plane
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / output_filename

        if not out_path.exists():
            shutil.copy2(source_path, out_path)

        row = {
            "ID": output_filename.replace(".nii.gz", ""),
            "plane": plane,
            "study_id": file_info["study_id"],
            "patient_id_full": file_info.get("patient_id_full", ""),
            "version": file_info["version"],
            "original_filename": file_info["original_filename"],
            "nifti_path": str(out_path),
        }

        if csv_key in csv_metadata:
            csv_row = csv_metadata[csv_key]
            for col, val in csv_row.items():
                if col not in ["csv_key", "NII_FileName"] and col not in row:
                    row[col] = val
            matched_count += 1
        else:
            unmatched_count += 1

        metadata_rows.append(row)
        planes.add(plane)

    logger.info(f"Matched {matched_count} files with CSV metadata, {unmatched_count} unmatched")

    logger.info("Step 4: Creating train.csv")
    df_all = pd.DataFrame(metadata_rows)
    priority_cols = ["ID", "plane", "study_id", "patient_id_full", "version", "original_filename"]
    other_cols = [c for c in df_all.columns if c not in priority_cols]
    df_all = df_all[priority_cols + other_cols]

    csv_path = save_dir / "train.csv"
    df_all.to_csv(csv_path, index=False)
    logger.info(f"Saved {len(df_all)} entries to {csv_path}")

    logger.info(f"Preprocessing complete. Total files: {len(files)}, Planes: {sorted(planes)}")
    for plane in sorted(planes):
        logger.info(f"  train/{plane}: {len(df_all[df_all['plane'] == plane])} files")


if __name__ == "__main__":
    main()
