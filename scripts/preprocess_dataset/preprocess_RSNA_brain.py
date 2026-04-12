"""
Preprocess RSNA-MICCAI Brain Tumor Radiogenomic Classification dataset for MedSliM.

Converts per-modality DICOM series to NIfTI.  Each subject has four structural
MRI modalities (FLAIR, T1w, T1wCE, T2w) stored as DICOM in separate folders.

Source layout (Kaggle download):
  {data_dir}/train/{subject_id}/{modality}/Image-*.dcm
  {data_dir}/test/{subject_id}/{modality}/Image-*.dcm
  {data_dir}/train_labels.csv          # BraTS21ID,MGMT_value

Output layout:
  {save_dir}/{subject_id}/{modality}.nii.gz + metadata.csv

Reference: https://www.kaggle.com/competitions/rsna-miccai-brain-tumor-radiogenomic-classification
"""
import argparse
import logging
import sys
import pydicom
import SimpleITK as sitk
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool

logger = logging.getLogger(__name__)

MODALITIES = ["FLAIR", "T1w", "T1wCE", "T2w"]


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
            subject_id = subject_dir.name
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


def convert_series_to_nifti(task: dict) -> dict | None:
    series_dir = Path(task["series_dir"])
    save_dir = Path(task["save_dir"])
    subject_id = task["subject_id"]
    modality = task["modality"]

    out_dir = save_dir / subject_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{modality}.nii.gz"

    if out_path.exists():
        logger.debug(f"Skipping (exists): {subject_id}/{modality}")
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

        dcm_file = next(series_dir.glob("*.dcm"), None)
        md = {}
        if dcm_file:
            ds = pydicom.dcmread(str(dcm_file), stop_before_pixels=True)
            md = dataset2dict(ds)

        md.update({
            "nifti_path": str(out_path),
            "subject_id": subject_id,
            "modality": modality,
            "split": task["split"],
            "size": list(img.GetSize()),
            "spacing": list(img.GetSpacing()),
        })
        logger.info(f"Converted: {subject_id}/{modality}")
        return md
    except Exception as e:
        logger.warning(f"Failed converting {subject_id}/{modality}: {e}")
        return None


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

    # Merge with MGMT labels for training subjects
    labels_csv = data_dir / "train_labels.csv"
    if labels_csv.exists() and not df_meta.empty:
        df_labels = pd.read_csv(labels_csv, dtype={"BraTS21ID": str})
        df_labels["BraTS21ID"] = df_labels["BraTS21ID"].str.zfill(5)
        df_meta = df_meta.merge(
            df_labels, left_on="subject_id", right_on="BraTS21ID",
            how="left", suffixes=("", "_label"),
        )
        df_meta.drop(columns=["BraTS21ID"], errors="ignore", inplace=True)
        logger.info(
            f"Merged MGMT labels: "
            f"{df_meta['MGMT_value'].notna().sum()} of {len(df_meta)} rows have labels"
        )

    df_meta.to_csv(save_dir / "metadata.csv", index=False)

    num_files = sum(1 for _ in save_dir.rglob("*.nii.gz"))
    logger.info(f"Finished. NIfTI files written: {num_files}")
    logger.info("=" * 60)
    logger.info("Preprocessing completed.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
