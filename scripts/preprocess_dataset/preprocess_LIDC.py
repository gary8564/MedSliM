"""
Preprocess LIDC-IDRI dataset.

Two independent steps (select with --convert-nifti, --process-labels, --organize-splits;
run all three when no step flag is given):

  1. NIfTI conversion (--convert-nifti): one volume per pylidc `Scan` (DICOM series -> NIfTI),
     written to `{save_dir}/volumes/{scan_id}.nii.gz`, with a `metadata.csv`
     kept for image inspection/debugging.
  2. Label processing (--process-labels): per-annotation `annotation.csv` from pylidc, then
     scan-level train/test annotation CSVs from per-nodule malignancy ratings.
  3. Train/test split (--organize-splits): move labeled volumes from `volumes/` into
     `{save_dir}/{split}/axial/{scan_id}.nii.gz` using `malignancy_labels.csv`.

Malignancy aggregation (nodule -> scan) follows the "uncertain case removal" strategy from Causey et al. 2018 (https://doi.org/10.1038/s41598-018-27569-w), as also implemented by https://github.com/mueller-franzes/MST/blob/main/scripts/preprocessing/lidc/step3_create_split.py:
  - For every nodule (a cluster of overlapping radiologist annotations,
    via `scan.cluster_annotations()`), export all radiologist ratings and
    nodule metadata to `annotation.csv`, following MST step2_export_labels.py
    (without segmentation-mask export, which MedSliM's whole-scan evaluation
    does not consume).
  - Clusters with more than 4 annotations are treated as "too close" /
    ambiguous grouping cases and excluded before label aggregation. This is
    the automatic pylidc failure mode behind the paper's "Nodules too close"
    exclusion.
  - For every remaining nodule, average the `malignancy` rating (1-5)
    across radiologists and round to the nearest integer.
  - Nodules whose rounded rating is 3 ("Indeterminate") are dropped.

Two label CSV variants are produced per split (train/test):
  - `{split}_binary.csv`:      ID, Malignant   (0 = benign (malignancy < 3), 1 = malignant (malignancy > 3);
  - `{split}_multiclass.csv`:  ID, malignancy  (0..3, remapped from the
                                                informative ratings {1,2,4,5} (Highly Unlikely, Moderately Unlikely, Moderately Suspicious, Highly Suspicious))

Train/test split using `StratifiedGroupKFold`.

Output layout:
  {save_dir}/volumes/{scan_id}.nii.gz   
  {save_dir}/{split}/axial/{scan_id}.nii.gz
  {save_dir}/metadata.csv
  {save_dir}/annotation.csv
  {save_dir}/nodule_labels.csv
  {save_dir}/malignancy_labels.csv
  {save_dir}/train_binary.csv, test_binary.csv
  {save_dir}/train_multiclass.csv, test_multiclass.csv
"""
import argparse
import logging
import shutil
import sys
import pandas as pd
import pydicom
import torch
import numpy as np

# pylidc 0.2.3 still uses NumPy 1.x aliases removed in NumPy 2.0 (e.g. np.int in
# Contour.to_matrix). Restore them before importing pylidc.
for _alias, _replacement in {
    "int": int,
    "float": float,
    "bool": bool,
    "object": object,
    "complex": complex,
}.items():
    if not hasattr(np, _alias):
        setattr(np, _alias, _replacement)

import pylidc as pl
import torchio as tio
import SimpleITK as sitk
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool
from functools import partial

logger = logging.getLogger(__name__)

ANNOTATION_LABELS = [
    "subtlety",
    "internalStructure",
    "calcification",
    "sphericity",
    "margin",
    "lobulation",
    "spiculation",
    "texture",
    "malignancy",
]

MALIGNANCY_MULTICLASS_MAP = {1: 0, 2: 1, 4: 2, 5: 3}
MALIGNANCY_CLASS_NAMES = {
    0: "Highly Unlikely",
    1: "Moderately Unlikely",
    2: "Moderately Suspicious",
    3: "Highly Suspicious",
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

def maybe_convert(x):
    if isinstance(x, pydicom.sequence.Sequence):
        # return [maybe_convert(item) for item in x]
        return None # Don't store this type of data 
    elif isinstance(x, pydicom.dataset.Dataset):  
        # return dataset2dict(x)
        return None # Don't store this type of data 
    elif isinstance(x, pydicom.multival.MultiValue):
        return list(x)
    elif isinstance(x, pydicom.valuerep.PersonName):
        return str(x)
    else:
        return x 


def dataset2dict(ds, exclude=['PixelData', '']):
    return {keyword:value for key in ds.keys() 
            if ((keyword := ds[key].keyword) not in exclude)  and ((value := maybe_convert(ds[key].value)) is not None) }

def reset_split_volumes_to_staging(save_dir: Path, plane: str = "axial") -> int:
    """Move NIfTIs from train/{plane}/ and test/{plane}/ back into volumes/ before re-splitting."""
    dest_dir = save_dir / "volumes"
    dest_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for split_name in ("train", "test"):
        split_plane_dir = save_dir / split_name / plane
        if not split_plane_dir.exists():
            continue
        for path in sorted(split_plane_dir.glob("*.nii.gz")):
            dest = dest_dir / path.name
            if dest.exists():
                path.unlink()
                continue
            shutil.move(str(path), str(dest))
            moved += 1
    return moved


def resolve_volume_path(scan_id: int, save_dir: Path) -> Path | None:
    """Return the on-disk NIfTI for a scan_id under volumes/."""
    path = save_dir / "volumes" / f"{scan_id}.nii.gz"
    return path if path.exists() else None


def clear_split_nifti_dirs(save_dir: Path, plane: str = "axial") -> None:
    """Remove all NIfTI files under train/{plane}/ and test/{plane}/ before reorganizing."""
    for split_name in ("train", "test"):
        split_plane_dir = save_dir / split_name / plane
        if not split_plane_dir.exists():
            continue
        for path in split_plane_dir.glob("*.nii.gz"):
            path.unlink()
        logger.info(f"Cleared {split_name}/{plane}/")


def organize_split_volumes(df_labels: pd.DataFrame, save_dir: Path, plane: str = "axial") -> None:
    """
    Move labeled scans from `volumes/` into `{split}/{plane}/`.

    Only scans present in `df_labels` with a split assignment are moved.
    Unlabeled / dropped scans remain in `volumes/`.
    """
    n_reset = reset_split_volumes_to_staging(save_dir, plane)
    if n_reset:
        logger.info(f"Reset {n_reset} NIfTI files from split dirs back into volumes/.")

    clear_split_nifti_dirs(save_dir, plane)

    for split_name in ["train", "test"]:
        df_split = df_labels[df_labels["split"] == split_name]
        if df_split.empty:
            logger.warning(f"No rows for split '{split_name}'; skipping {split_name}/{plane}/")
            continue

        split_plane_dir = save_dir / split_name / plane
        split_plane_dir.mkdir(parents=True, exist_ok=True)

        n_moved = n_missing = 0
        for scan_id in df_split["scan_id"]:
            src = resolve_volume_path(int(scan_id), save_dir)
            dst = split_plane_dir / f"{scan_id}.nii.gz"
            if src is None:
                n_missing += 1
                continue
            if src.resolve() != dst.resolve():
                shutil.move(str(src), str(dst))
            n_moved += 1

        msg = f"{split_name}/{plane}/: moved {n_moved}/{len(df_split)} scans"
        if n_missing:
            msg += f" ({n_missing} missing in volumes/)"
        logger.info(msg)


def scan2nifti(scan_id, save_dir):
    # Get DICOMs in correct order (PyLIDC fixes duplicate z and sorts by z coordinate)
    scan = pl.query(pl.Scan).filter(pl.Scan.id == scan_id).first()
    images = scan.load_all_dicom_images()

    # Get path to series
    path_series = Path(scan.get_path_to_dicom_files())
    
    # In-plane pixel spacing (mm) — LIDC uses square pixels
    row_spacing = col_spacing = scan.pixel_spacing
    
    # Through-plane spacing (mm)
    slice_spacing = float(abs(scan.slice_spacing))
    
    # Image position and orientation (row and column direction cosines)
    image_position = np.array(images[0].ImagePositionPatient, float)
    image_orientation = np.array(images[0].ImageOrientationPatient, float)
    row_cosines = image_orientation[:3]   # direction of increasing column index (along a row)
    col_cosines = image_orientation[3:]   # direction of increasing row index (down a column)
    slice_cosines = np.cross(row_cosines, col_cosines)

    # Construct affine in DICOM (LPS) coordinate system.
    # Volume from to_volume() has shape (Rows, Cols, Slices):
    #   axis 0 (row index)    → moves in column direction → col_cosines
    #   axis 1 (column index) → moves in row direction    → row_cosines
    affine = np.eye(4)
    affine[:3, 0] = col_cosines * row_spacing
    affine[:3, 1] = row_cosines * col_spacing
    affine[:3, 2] = slice_cosines * slice_spacing
    affine[:3, 3] = image_position

    # Convert from DICOM LPS to NIfTI RAS: negate x (L→R) and y (P→A)
    affine[0, :] *= -1
    affine[1, :] *= -1

    # Return the scan as a 3D numpy array volume and affine transform to world coordinates
    img_vol = scan.to_volume()
    img_tio = tio.ScalarImage(tensor=img_vol[None], affine=affine)

    # Read Metadata 
    ds = pydicom.dcmread(next(path_series.glob('*.dcm'), None), stop_before_pixels=True)
    metadata = dataset2dict(ds)

    # Write to NIfTI file under volumes/
    out_dir = save_dir / "volumes"
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f'{scan_id}.nii.gz'
    logger.info(f"Writing file: {filename}:")
    img_tio.save(out_dir / filename)

    # Add additional information 
    metadata['size'] = list(img_tio.spatial_shape)
    metadata['nifti_path'] = str(out_dir / filename)
    metadata['scan_id'] = scan_id
    return metadata


def scan2annotation_rows(scan_id: int) -> list[dict]:
    """Exportper-radiologist annotation rows for one pylidc scan."""
    scan = pl.query(pl.Scan).filter(pl.Scan.id == scan_id).first()

    rows = []
    for nodule_idx, nodule in enumerate(scan.cluster_annotations(verbose=False)):
        annotation_num = len(nodule)
        for annotation_idx, ann in enumerate(nodule):
            row = {label: getattr(ann, label) for label in ANNOTATION_LABELS}
            row.update({
                "bbox": [[d.start, d.stop] for d in ann.bbox()],
                "scan_id": scan.id,
                "nodule_idx": nodule_idx,
                "annotation_idx": annotation_idx,
                "annotation_num": annotation_num,
                "annotation_id": ann.id,
                "patient_id": scan.patient_id,
                "study_instance_uid": scan.study_instance_uid,
                "series_instance_uid": scan.series_instance_uid,
                # pylidc can occasionally return clusters with >4 annotations
                # when nearby nodules cannot be separated automatically.
                "too_close": annotation_num > 4,
            })
            rows.append(row)
    return rows


def collect_annotation_rows(num_scans: int, workers: int) -> pd.DataFrame:
    """Query pylidc directly for per-annotation labels."""
    scan_ids = range(1, num_scans + 1)
    rows = []
    with Pool(processes=max(1, int(workers))) as pool:
        for scan_rows in tqdm(pool.imap_unordered(scan2annotation_rows, scan_ids), total=len(scan_ids), desc="Exporting annotations"):
            rows.extend(scan_rows)

    df_annotations = pd.DataFrame(rows)
    logger.info(
        f"Exported {len(df_annotations)} radiologist annotation rows from "
        f"{df_annotations['scan_id'].nunique() if not df_annotations.empty else 0} scans."
    )
    return df_annotations


def build_malignancy_labels(df_annotations: pd.DataFrame, num_scans: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate nodule level labels to one whole-scan label.
    """
    if df_annotations.empty:
        return pd.DataFrame(), pd.DataFrame()

    unique_cols = [
        "patient_id",
        "study_instance_uid",
        "series_instance_uid",
        "scan_id",
        "nodule_idx",
    ]

    df_too_close = df_annotations[df_annotations["annotation_num"] > 4]
    too_close_nodules = (
        df_too_close[unique_cols].drop_duplicates()
        if not df_too_close.empty else pd.DataFrame(columns=unique_cols)
    )
    if not too_close_nodules.empty:
        logger.info(
            f"Dropping {len(too_close_nodules)} nodule clusters with >4 annotations "
            "(ambiguous/too-close automatic grouping)."
        )

    df_clean = df_annotations[df_annotations["annotation_num"] <= 4].copy()
    if df_clean.empty:
        logger.warning("All annotated nodule clusters were marked too-close/ambiguous; no usable labels remain.")
        return pd.DataFrame(), pd.DataFrame()

    df_nodule_scores = (
        df_clean.groupby(unique_cols)["malignancy"]
        .apply(lambda x: int(x.mean().round()))
        .reset_index()
    )
    df_nodule_meta = df_clean.drop_duplicates(unique_cols).drop(columns="malignancy")
    df_nodules = pd.merge(df_nodule_scores, df_nodule_meta, on=unique_cols).reset_index(drop=True)

    n_score3 = int((df_nodules["malignancy"] == 3).sum())
    if n_score3:
        logger.info(f"Dropping {n_score3} nodules with rounded malignancy score 3 (Indeterminate).")
    df_nodules = df_nodules[df_nodules["malignancy"] != 3].copy()
    df_nodules["Malignant"] = (df_nodules["malignancy"] > 3).astype(int)

    if df_nodules.empty:
        return df_nodules, pd.DataFrame()

    # MedSliM evaluates whole CT volumes, so collapse multiple informative
    # nodules in one scan to the worst remaining malignancy rating.
    too_close_scan_ids = set(too_close_nodules["scan_id"].astype(int).tolist())
    df_scan = (
        df_nodules.groupby(["scan_id", "patient_id"])
        .agg(
            num_informative_nodules=("nodule_idx", "nunique"),
            malignancy=("malignancy", "max"),
        )
        .reset_index()
    )
    df_scan["Malignant"] = (df_scan["malignancy"] > 3).astype(int)
    df_scan["has_too_close_nodule"] = df_scan["scan_id"].isin(too_close_scan_ids)

    n_dropped = num_scans - len(df_scan)
    logger.info(
        f"Scans with an informative non-indeterminate nodule: {len(df_scan)}/{num_scans} "
        f"({n_dropped} dropped: Annotated nodule <3mm / too-close clusters / every usable nodule was rated 3/Indeterminate)."
    )
    return df_nodules, df_scan


def split_by_patient(df_labels: pd.DataFrame, test_fraction: float, seed: int) -> pd.DataFrame:
    """Patient-grouped, Malignant-stratified train/test split (no patient leakage)."""
    from sklearn.model_selection import StratifiedGroupKFold

    n_splits = max(2, round(1.0 / test_fraction))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    _, test_idx = next(sgkf.split(df_labels, df_labels["Malignant"], groups=df_labels["patient_id"]))

    df_labels = df_labels.reset_index(drop=True).copy()
    df_labels["split"] = "train"
    df_labels.loc[test_idx, "split"] = "test"

    n_train_patients = df_labels.loc[df_labels["split"] == "train", "patient_id"].nunique()
    n_test_patients = df_labels.loc[df_labels["split"] == "test", "patient_id"].nunique()
    logger.info(
        f"Split into {len(df_labels[df_labels['split'] == 'train'])} train / "
        f"{len(df_labels[df_labels['split'] == 'test'])} test scans "
        f"({n_train_patients} / {n_test_patients} unique patients)."
    )
    return df_labels


def write_annotation_csvs(df_labels: pd.DataFrame, save_dir: Path, df_nodules: pd.DataFrame | None = None) -> None:
    df_labels = df_labels.copy()
    df_labels["ID"] = df_labels["scan_id"].astype(str)
    df_labels["malignancy_class"] = df_labels["malignancy"].map(MALIGNANCY_MULTICLASS_MAP)

    for split_name in ["train", "test"]:
        df_split = df_labels[df_labels["split"] == split_name]
        if df_split.empty:
            logger.warning(f"No rows found for split '{split_name}'; skipping {split_name}_binary.csv/{split_name}_multiclass.csv")
            continue

        df_binary = df_split[["ID", "Malignant"]].copy()
        df_binary.to_csv(save_dir / f"{split_name}_binary.csv", index=False)
        counts_b = df_binary["Malignant"].value_counts().sort_index()
        logger.info(
            f"{split_name}_binary.csv written with {len(df_binary)} entries. "
            f"Malignant distribution (0=benign, 1=malignant): {counts_b.to_dict()}"
        )

        df_multiclass = df_split[["ID", "malignancy_class"]].rename(columns={"malignancy_class": "malignancy"}).copy()
        df_multiclass.to_csv(save_dir / f"{split_name}_multiclass.csv", index=False)
        counts_m = df_multiclass["malignancy"].value_counts().sort_index()
        counts_m_named = {MALIGNANCY_CLASS_NAMES[k]: v for k, v in counts_m.to_dict().items()}
        logger.info(
            f"{split_name}_multiclass.csv written with {len(df_multiclass)} entries. "
            f"malignancy distribution: {counts_m_named}"
        )

    if df_nodules is not None:
        df_nodules.to_csv(save_dir / "nodule_labels.csv", index=False)
        logger.info(f"Per-nodule label provenance written to nodule_labels.csv ({len(df_nodules)} nodules).")

    df_labels.to_csv(save_dir / "malignancy_labels.csv", index=False)
    logger.info(f"Full per-scan label provenance written to malignancy_labels.csv ({len(df_labels)} scans).")


def load_malignancy_labels(save_dir: Path) -> pd.DataFrame:
    labels_path = save_dir / "malignancy_labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(
            f"{labels_path} not found. Run with --process-labels first."
        )
    df_labels = pd.read_csv(labels_path)
    if "split" not in df_labels.columns:
        raise ValueError(
            f"{labels_path} has no 'split' column. Re-run with --process-labels."
        )
    return df_labels


def run_convert_nifti(save_dir: Path, workers: int, scan_ids: range) -> None:
    logger.info("=" * 60)
    logger.info("Step: Converting DICOM scans to NIfTI ...")
    logger.info("=" * 60)
    metadata_list = []
    scan2nifti_with_dir = partial(scan2nifti, save_dir=save_dir)
    with Pool(processes=max(1, int(workers))) as pool:
        for meta in tqdm(pool.imap_unordered(scan2nifti_with_dir, scan_ids), total=len(scan_ids)):
            metadata_list.append(meta)

    df = pd.DataFrame(metadata_list)
    df.to_csv(save_dir / "metadata.csv", index=False)
    num_files = len(list((save_dir / "volumes").glob("*.nii.gz")))
    logger.info(f"Finished. NIfTI files written to volumes/: {num_files}")


def run_process_labels(
    save_dir: Path, workers: int, num_scans: int, test_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    logger.info("=" * 60)
    logger.info("Step: Exporting per-annotation malignancy labels from pylidc ...")
    logger.info("=" * 60)
    df_annotations = collect_annotation_rows(num_scans, workers)
    df_annotations.to_csv(save_dir / "annotation.csv", index=False)
    logger.info(f"Per-annotation labels written to annotation.csv ({len(df_annotations)} rows).")

    logger.info("=" * 60)
    logger.info("Step: Building nodule-level and scan-level malignancy labels ...")
    logger.info("=" * 60)
    df_nodules, df_labels = build_malignancy_labels(df_annotations, num_scans)
    if df_labels.empty:
        logger.warning("No scans with informative nodules found; skipping label CSV generation.")
        return df_nodules, df_labels

    df_labels = split_by_patient(df_labels, test_fraction, seed)

    logger.info("=" * 60)
    logger.info("Step: Writing train/test annotation CSVs ...")
    logger.info("=" * 60)
    write_annotation_csvs(df_labels, save_dir, df_nodules)
    return df_nodules, df_labels


def run_organize_splits(save_dir: Path, df_labels: pd.DataFrame | None = None) -> None:
    if df_labels is None:
        df_labels = load_malignancy_labels(save_dir)

    logger.info("=" * 60)
    logger.info("Step: Organizing volumes/ into train/axial and test/axial ...")
    logger.info("=" * 60)
    organize_split_volumes(df_labels, save_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess LIDC-IDRI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  Full pipeline (default):\n"
            "    python preprocess_LIDC.py --save-dir /path/to/out\n"
            "  Labels only:\n"
            "    python preprocess_LIDC.py --save-dir /path/to/out --process-labels\n"
            "  Re-organize volumes after labels exist:\n"
            "    python preprocess_LIDC.py --save-dir /path/to/out --organize-splits\n"
        ),
    )
    parser.add_argument("--data-dir", type=str, default="/hpcwork/rwth1833/datasets/LIDC-IDRI/TCIA_LIDC-IDRI_20200921/LIDC-IDRI", help="Root folder containing the LIDC-IDRI dataset")
    parser.add_argument("--save-dir", type=str, default="/hpcwork/rwth1833/datasets/preprocessed/LIDC-IDRI", help="Output directory for NIfTI and metadata")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--convert-nifti", action="store_true", help="Convert DICOM scans to NIfTI under volumes/.")
    parser.add_argument("--process-labels", action="store_true", help="Export malignancy labels and write train/test CSVs.")
    parser.add_argument("--organize-splits", action="store_true", help="Move labeled volumes from volumes/ into train/axial and test/axial.")
    parser.add_argument("--test-fraction", type=float, default=0.2, help="Fraction of patients held out for the test split")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the train/test split")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    run_all = not (args.convert_nifti or args.process_labels or args.organize_splits)
    convert_nifti = args.convert_nifti or run_all
    process_labels = args.process_labels or run_all
    organize_splits = args.organize_splits or run_all

    setup_logging(args.verbose)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    num_scans = len(list(pl.query(pl.Scan)))
    scan_ids = range(1, num_scans + 1)

    if convert_nifti:
        run_convert_nifti(save_dir, args.workers, scan_ids)

    df_labels: pd.DataFrame | None = None
    if process_labels:
        _, df_labels = run_process_labels(
            save_dir, args.workers, num_scans, args.test_fraction, args.seed
        )

    if organize_splits:
        if df_labels is not None and not df_labels.empty:
            run_organize_splits(save_dir, df_labels)
        elif df_labels is not None and df_labels.empty:
            logger.warning("Skipping --organize-splits because no labeled scans were found.")
        else:
            run_organize_splits(save_dir)

    logger.info("=" * 60)
    logger.info("Preprocessing completed.")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
