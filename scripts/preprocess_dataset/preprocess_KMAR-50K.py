#TODO: the scripts for naming convention needs to be merged into the whole preprocessing script instead of --rename flag when publishing. 
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
    
    # Remove .nii.gz extension
    stem = filename.replace(".nii.gz", "")
    
    # More flexible pattern:
    # - year: 4 digits
    # - patient_id: anything until we hit the plane
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
    
    # Normalize patient_id for study_id:
    # Extract just the numeric prefix before any hyphen or X characters for cleaner IDs
    # e.g., "144-20210515001368" → "144"
    #       "383XX20221031001450" → "383"
    # But keep the full ID for uniqueness in output filename
    id_match = re.match(r"^(\d+)", patient_id)
    short_id = id_match.group(1) if id_match else patient_id
    
    # study_id uses year + short numeric ID for grouping
    study_id = f"{year}_{short_id}"
    
    version_int = version.split(".")[0]
    output_filename = f"{study_id}_MR{version_int}.nii.gz"
    
    # Original filename key for CSV matching (try multiple formats)
    # CSV may use different formats, so we generate potential keys
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
    Load and merge TrainingCohort.csv and TestingCohort.csv.
    
    Creates a unified DataFrame with a common key for matching with NIfTI files.
    """
    dfs = []
    
    # Load training CSV (has NII_FileName column)
    if train_csv.exists():
        df_train = pd.read_csv(train_csv)
        # Clean up the NII_FileName column and use as key
        df_train["csv_key"] = df_train["NII_FileName"].apply(
            lambda x: str(x).strip().strip('"').strip("'") if pd.notna(x) else ""
        )
        df_train["original_split"] = "train"
        dfs.append(df_train)
        logger.info(f"Loaded {len(df_train)} entries from TrainingCohort.csv")
    
    # Load testing CSV (has Patient column instead of NII_FileName)
    if test_csv.exists():
        df_test = pd.read_csv(test_csv)
        # Patient format: "2023_34_coronal_0" - need to add .nii.gz
        df_test["csv_key"] = df_test["Patient"].apply(
            lambda x: str(x).strip().strip('"').strip("'") + ".nii.gz" if pd.notna(x) else ""
        )
        # Rename Patient to NII_FileName for consistency
        df_test = df_test.rename(columns={"Patient": "NII_FileName"})
        df_test["original_split"] = "test"
        dfs.append(df_test)
        logger.info(f"Loaded {len(df_test)} entries from TestingCohort.csv")
    
    if not dfs:
        logger.warning("No CSV files found!")
        return pd.DataFrame()
    
    # Merge all CSVs
    df_merged = pd.concat(dfs, ignore_index=True)
    logger.info(f"Merged CSV has {len(df_merged)} total entries")
    
    return df_merged


def build_rename_map(csv_path: Path) -> tuple[dict[tuple[str, str], str], pd.DataFrame]:
    df = pd.read_csv(csv_path)
    rename_map: dict[tuple[str, str], str] = {}
    collision_check: dict[tuple[str, str], str] = {}  # (new_id, plane) -> old_id

    for _, row in df.iterrows():
        old_id = str(row["ID"])
        study_id = str(row["study_id"])
        version = str(row["version"])
        plane = str(row["plane"])
        version_int = version.split(".")[0]  # "0.0" -> "0", "1.0" -> "1"
        new_id = f"{study_id}_MR{version_int}"

        # Collision detection
        key = (new_id, plane)
        if key in collision_check and collision_check[key] != old_id:
            raise ValueError(
                f"Naming collision: '{collision_check[key]}' and '{old_id}' "
                f"both map to '{new_id}' in plane '{plane}'"
            )
        collision_check[key] = old_id
        rename_map[(old_id, plane)] = new_id

    n_changed = sum(1 for (old, _), new in rename_map.items() if old != new)
    logger.info(f"Built rename map: {len(rename_map)} entries, {n_changed} will change")

    return rename_map, df


def rename_to_medslim_convention(
    nifti_base_dir: Path,
    feat_cache_dir: Path | None,
    csv_path: Path,
    dry_run: bool = False,
) -> None:
    rename_map, df = build_rename_map(csv_path)

    logger.info("Renaming NIfTI files...")
    renamed, skipped, missing = 0, 0, 0
    for (old_id, plane), new_id in rename_map.items():
        if old_id == new_id:
            skipped += 1
            continue
        old_path = nifti_base_dir / "train" / plane / f"{old_id}.nii.gz"
        new_path = nifti_base_dir / "train" / plane / f"{new_id}.nii.gz"
        if old_path.exists():
            if not dry_run:
                old_path.rename(new_path)
            logger.debug(f"  {plane}/{old_path.name} -> {new_path.name}")
            renamed += 1
        elif new_path.exists():
            skipped += 1  # already renamed
        else:
            logger.warning(f"  NIfTI not found: {old_path}")
            missing += 1
    logger.info(f"  NIfTI: {renamed} renamed, {skipped} unchanged/done, {missing} missing")

    if feat_cache_dir and feat_cache_dir.exists():
        logger.info("Renaming safetensors files...")
        model_dirs = sorted([d for d in feat_cache_dir.iterdir() if d.is_dir()])
        for model_dir in model_dirs:
            renamed_st, skipped_st, missing_st = 0, 0, 0
            for (old_id, plane), new_id in rename_map.items():
                if old_id == new_id:
                    skipped_st += 1
                    continue
                old_path = model_dir / "train" / plane / f"{old_id}.safetensors"
                new_path = model_dir / "train" / plane / f"{new_id}.safetensors"
                if old_path.exists():
                    if not dry_run:
                        old_path.rename(new_path)
                    renamed_st += 1
                elif new_path.exists():
                    skipped_st += 1
                else:
                    missing_st += 1
            logger.info(
                f"{model_dir.name}: {renamed_st} renamed, "
                f"{skipped_st} unchanged/done, {missing_st} missing"
            )
    elif feat_cache_dir:
        logger.warning(f"Feature cache dir not found: {feat_cache_dir}")

    # Build a simple old_id to new_id map for CSV update
    id_map = {old_id: new_id for (old_id, _), new_id in rename_map.items()}

    if not dry_run:
        logger.info("Updating train.csv...")
        df["ID"] = df["ID"].map(lambda x: id_map.get(str(x), str(x)))
        if "nifti_path" in df.columns:
            df["nifti_path"] = df.apply(
                lambda row: str(
                    nifti_base_dir / "train" / str(row["plane"]) / f"{row['ID']}.nii.gz"
                ),
                axis=1,
            )
        df.to_csv(csv_path, index=False)
        logger.info(f"  Updated {csv_path}")
    else:
        logger.info("[DRY RUN] No files were modified.")


def main():
    parser = argparse.ArgumentParser(description="Preprocess KMAR-50K dataset for SSL pretraining")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/hpcwork/rwth1833/datasets/KMAR-50K",
        help="Root folder containing KMAR-50K dataset",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="/hpcwork/rwth1833/datasets/preprocessed/KMAR-50K",
        help="Output directory for organized NIfTI files",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--rename", action="store_true",
        help="Rename existing files to MedSliM-compatible naming convention "
             "({study_id}_MR{version_int}) instead of running preprocessing",
    )
    parser.add_argument(
        "--feat-cache-dir", type=str, default=None,
        help="Path to precomputed feature cache dir for renaming safetensors "
             "(e.g., /hpcwork/rwth1833/feat_caches/KMAR-50K/slices_raw/adaptive)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be renamed without actually modifying files",
    )
    args = parser.parse_args()

    setup_logging(args.verbose)

    save_dir = Path(args.save_dir)

    if args.rename:
        csv_path = save_dir / "train.csv"
        if not csv_path.exists():
            logger.error(f"CSV not found: {csv_path}. Run preprocessing first.")
            return
        feat_cache_dir = Path(args.feat_cache_dir) if args.feat_cache_dir else None
        rename_to_medslim_convention(
            nifti_base_dir=save_dir,
            feat_cache_dir=feat_cache_dir,
            csv_path=csv_path,
            dry_run=args.dry_run,
        )
        return

    data_root = Path(args.data_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Define GroundTruthData directories
    ground_truth_dirs = [
        data_root / "KMAR-50K" / "GroundTruthData_part1",
        data_root / "KMAR-50K-part2" / "GroundTruthData_part2",
        data_root / "KMAR-50K" / "Testing _GroundTruthData" / "1_GroundTruthData",
    ]

    # CSV files
    train_csv = data_root / "KMAR-50K" / "TrainingCohort.csv"
    test_csv = data_root / "KMAR-50K" / "TestingCohort.csv"

    logger.info("=" * 60)
    logger.info("Step 1: Collecting GroundTruthData NIfTI files")
    logger.info("=" * 60)
    
    files = collect_nifti_files(ground_truth_dirs)
    
    if not files:
        logger.error("No NIfTI files found!")
        return

    logger.info("=" * 60)
    logger.info("Step 2: Loading and merging CSV metadata")
    logger.info("=" * 60)
    
    df_csv = load_and_merge_csvs(train_csv, test_csv)
    
    # Create csv_key to metadata mapping
    csv_metadata = {}
    if not df_csv.empty:
        for _, row in df_csv.iterrows():
            key = row.get("csv_key", "")
            if key:
                csv_metadata[key] = row.to_dict()

    logger.info("=" * 60)
    logger.info("Step 3: Organizing files into train/{plane}/ structure")
    logger.info("=" * 60)
    
    # All files go to train/ for SSL pretraining
    metadata_rows = []
    planes = set()
    matched_count = 0
    unmatched_count = 0
    
    for file_info in tqdm(files, desc="Organizing files"):
        plane = file_info["plane"]
        output_filename = file_info["output_filename"]
        source_path = Path(file_info["source_path"])
        csv_key = file_info["csv_key"]
        
        # Create output directory: train/{plane}/
        out_dir = save_dir / "train" / plane
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Output path
        out_path = out_dir / output_filename
        
        # Copy file to output directory
        if not out_path.exists():
            shutil.copy2(source_path, out_path)
        
        # Build metadata row
        row = {
            "ID": output_filename.replace(".nii.gz", ""),
            "plane": plane,
            "study_id": file_info["study_id"],
            "patient_id_full": file_info.get("patient_id_full", ""),
            "version": file_info["version"],
            "original_filename": file_info["original_filename"],
            "nifti_path": str(out_path),
        }
        
        # Match with CSV metadata if available
        if csv_key in csv_metadata:
            csv_row = csv_metadata[csv_key]
            # Add CSV columns (excluding keys we already have)
            for col, val in csv_row.items():
                if col not in ["csv_key", "NII_FileName"] and col not in row:
                    row[col] = val
            matched_count += 1
        else:
            unmatched_count += 1
        
        metadata_rows.append(row)
        planes.add(plane)
    
    logger.info(f"Matched {matched_count} files with CSV metadata")
    logger.info(f"Unmatched files (no CSV entry): {unmatched_count}")

    logger.info("=" * 60)
    logger.info("Step 4: Creating train.csv")
    logger.info("=" * 60)
    
    df_all = pd.DataFrame(metadata_rows)
    
    # Reorder columns: ID and plane first, then others
    priority_cols = ["ID", "plane", "study_id", "patient_id_full", "version", "original_filename"]
    other_cols = [c for c in df_all.columns if c not in priority_cols]
    df_all = df_all[priority_cols + other_cols]
    
    # Save train.csv
    csv_path = save_dir / "train.csv"
    df_all.to_csv(csv_path, index=False)
    logger.info(f"Saved {len(df_all)} entries to {csv_path}")
    
    # Summary
    logger.info("=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info(f"Total files processed: {len(files)}")
    logger.info(f"Planes: {sorted(planes)}")
    
    for plane in sorted(planes):
        count = len(df_all[df_all["plane"] == plane])
        logger.info(f"  train/{plane}: {count} files")
    
    logger.info("=" * 60)
    logger.info("Preprocessing completed.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
