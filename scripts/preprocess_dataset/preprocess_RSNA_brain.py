"""
Preprocess RSNA-MICCAI Brain Tumor Radiogenomic Classification dataset for MedSliM.

Converts per-modality DICOM series to NIfTI.  Each subject has four structural
MRI modalities (FLAIR, T1w, T1wCE, T2w) stored as DICOM in separate folders.

Source layout (Kaggle download):
  {data_dir}/train/{subject_id}/{modality}/Image-*.dcm
  {data_dir}/test/{subject_id}/{modality}/Image-*.dcm
  {data_dir}/train_labels.csv          # BraTS21ID,MGMT_value

Output layout (MedSliM):
  {save_dir}/{flair,t1w,t1wce,t2w}/{train,test}/axial/{subject_id}.nii.gz
  {save_dir}/metadata.csv
  {save_dir}/train_labels.csv

Reference: https://www.kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification
"""
import argparse
import logging
import shutil
import sys
import pydicom
import SimpleITK as sitk
import nibabel as nib
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool

logger = logging.getLogger(__name__)

MODALITIES = ["FLAIR", "T1w", "T1wCE", "T2w"]
MODALITY_DIR = {
    "FLAIR": "flair",
    "T1w": "t1w",
    "T1wCE": "t1wce",
    "T2w": "t2w",
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
        return None
    elif isinstance(x, pydicom.dataset.Dataset):
        return None
    elif isinstance(x, pydicom.multival.MultiValue):
        return list(x)
    elif isinstance(x, pydicom.valuerep.PersonName):
        return str(x)
    else:
        return x


def dataset2dict(ds, exclude=['PixelData', '']):
    return {
        keyword: value
        for key in ds.keys()
        if ((keyword := ds[key].keyword) not in exclude)
        and ((value := maybe_convert(ds[key].value)) is not None)
    }


def normalize_subject_id(subject_id: str) -> str:
    sid = str(subject_id)
    return sid.zfill(5) if sid.isdigit() else sid


def output_path(save_dir: Path, split: str, modality: str, subject_id: str) -> Path:
    """MedSliM on-disk path: {modality}/{split}/axial/{subject_id}.nii.gz"""
    return (
        save_dir
        / MODALITY_DIR[modality]
        / split
        / "axial"
        / f"{normalize_subject_id(subject_id)}.nii.gz"
    )


def collect_tasks(data_dir: Path, save_dir: Path) -> list[dict]:
    """Walk train/ and test/ splits and collect one task per (subject, modality) pair."""
    tasks = []
    for split in ["train", "test"]:
        split_dir = data_dir / split
        if not split_dir.exists():
            logger.warning(f"Split directory not found: {split_dir}")
            continue

        subject_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        for subject_dir in subject_dirs:
            subject_id = normalize_subject_id(subject_dir.name)
            for modality in MODALITIES:
                modality_dir = subject_dir / modality
                if not modality_dir.exists():
                    logger.debug(f"Missing modality {modality} for {subject_id}")
                    continue

                dcm_files = list(modality_dir.glob("*.dcm"))
                if not dcm_files:
                    logger.debug(f"No DICOM files in {modality_dir}")
                    continue

                tasks.append({
                    "series_dir": str(modality_dir),
                    "save_dir": str(save_dir),
                    "subject_id": subject_id,
                    "modality": modality,
                    "split": split,
                })
    return tasks


def _metadata_from_dicom_and_nifti(
    series_dir: Path,
    out_path: Path,
    subject_id: str,
    modality: str,
    split: str,
    size: list | None = None,
    spacing: list | None = None,
) -> dict:
    md = {}
    dcm_file = next(series_dir.glob("*.dcm"), None)
    if dcm_file is not None:
        ds = pydicom.dcmread(str(dcm_file), stop_before_pixels=True)
        md = dataset2dict(ds)

    if size is None or spacing is None:
        img = nib.load(str(out_path))
        # nibabel shape is (i, j, k); SimpleITK GetSize is (x, y, z) in the same order here
        size = list(img.shape)
        spacing = list(img.header.get_zooms()[:3])

    md.update({
        "nifti_path": str(out_path),
        "subject_id": subject_id,
        "modality": modality,
        "split": split,
        "size": list(size),
        "spacing": list(spacing),
    })
    return md


def convert_series_to_nifti(task: dict) -> dict | None:
    series_dir = Path(task["series_dir"])
    save_dir = Path(task["save_dir"])
    subject_id = normalize_subject_id(task["subject_id"])
    modality = task["modality"]
    split = task["split"]
    out_path = output_path(save_dir, split, modality, subject_id)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        logger.debug(f"Skipping convert (exists): {out_path.relative_to(save_dir)}")
        try:
            return _metadata_from_dicom_and_nifti(
                series_dir, out_path, subject_id, modality, split
            )
        except Exception as e:
            logger.warning(f"Failed reading existing {out_path}: {e}")
            return None

    try:
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(series_dir))
        if not dicom_names:
            logger.warning(f"No DICOM series found in {series_dir}")
            return None
        reader.SetFileNames(dicom_names)
        img = reader.Execute()
        sitk.WriteImage(img, str(out_path))

        md = _metadata_from_dicom_and_nifti(
            series_dir,
            out_path,
            subject_id,
            modality,
            split,
            size=list(img.GetSize()),
            spacing=list(img.GetSpacing()),
        )
        logger.info(f"Converted: {out_path.relative_to(save_dir)}")
        return md
    except Exception as e:
        logger.warning(f"Failed converting {subject_id}/{modality}: {e}")
        return None


def merge_mgmt_labels(df_meta: pd.DataFrame, labels_csv: Path) -> pd.DataFrame:
    if not labels_csv.exists() or df_meta.empty:
        return df_meta
    df_labels = pd.read_csv(labels_csv, dtype={"BraTS21ID": str})
    df_labels["BraTS21ID"] = df_labels["BraTS21ID"].map(normalize_subject_id)
    df_meta = df_meta.copy()
    df_meta["subject_id"] = df_meta["subject_id"].map(normalize_subject_id)
    df_meta = df_meta.merge(
        df_labels, left_on="subject_id", right_on="BraTS21ID",
        how="left", suffixes=("", "_label"),
    )
    df_meta.drop(columns=["BraTS21ID"], errors="ignore", inplace=True)
    logger.info(
        f"Merged MGMT labels: "
        f"{df_meta['MGMT_value'].notna().sum()} of {len(df_meta)} rows have labels"
    )
    return df_meta


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess RSNA-MICCAI Brain Tumor Radiogenomic: DICOM to NIfTI"
    )
    parser.add_argument(
        "--data-dir", type=str,
        default="/hpcwork/rwth1833/datasets/RSNA_Brain_Tumor_Radiogenomic",
        help="Root folder containing train/, test/, and train_labels.csv",
    )
    parser.add_argument(
        "--save-dir", type=str,
        default="/hpcwork/rwth1833/datasets/preprocessed/RSNA_Brain_Tumor_Radiogenomic",
        help="Output directory for NIfTI files and metadata",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)

    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Collecting DICOM series tasks ...")
    tasks = collect_tasks(data_dir, save_dir)
    logger.info(f"Found {len(tasks)} (subject, modality) pairs to convert")

    split_counts = {}
    for t in tasks:
        split_counts[t["split"]] = split_counts.get(t["split"], 0) + 1
    for s, c in sorted(split_counts.items()):
        logger.info(f"  {s}: {c} series")

    logger.info("=" * 60)
    logger.info("Converting DICOM to NIfTI ...")
    logger.info("=" * 60)

    metadata_rows = []
    with Pool(processes=max(1, args.workers)) as pool:
        chunksize = max(1, len(tasks) // (max(1, args.workers) * 4))
        for md in tqdm(
            pool.imap_unordered(convert_series_to_nifti, tasks, chunksize=chunksize),
            total=len(tasks),
        ):
            if md is not None:
                metadata_rows.append(md)

    df_meta = pd.DataFrame(metadata_rows)
    labels_csv = data_dir / "train_labels.csv"
    df_meta = merge_mgmt_labels(df_meta, labels_csv)
    df_meta.to_csv(save_dir / "metadata.csv", index=False)

    # Keep label CSV beside the NIfTIs so raw can be deleted later.
    if labels_csv.exists():
        shutil.copy2(labels_csv, save_dir / "train_labels.csv")
        logger.info("Copied train_labels.csv into save_dir")

    num_files = sum(1 for _ in save_dir.rglob("*.nii.gz"))
    logger.info(f"Finished. NIfTI files on disk: {num_files}; metadata rows: {len(df_meta)}")
    if num_files and len(df_meta) != num_files:
        logger.warning(
            f"metadata rows ({len(df_meta)}) != NIfTI count ({num_files}). "
            "Some conversions may have failed."
        )
    logger.info("=" * 60)
    logger.info("Preprocessing completed.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
