"""
Preprocess UPENN-GBM dataset for MedSliM.

Converts DICOM structural brain MRI series to NIfTI using SimpleITK.
Filters to structural modalities only (T1 pre/post, T2, FLAIR);
skips DTI and perfusion series.

Uses the download metadata.csv to identify series and their modalities.

Source layout:
  {data_dir}/UPENN-GBM/{patient_id}/{study}/{series}/*.dcm

Output layout:
  {save_dir}/{modality}/{patient_id}_{study_idx}.nii.gz + metadata.csv
"""
import argparse
import logging
import re
import sys
import pydicom
import SimpleITK as sitk
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool

logger = logging.getLogger(__name__)

# Modality classification patterns (applied to Series Description, case-insensitive)
# Order matters: first match wins
MODALITY_RULES = [
    # Skip non-structural sequences first
    ("_skip", re.compile(r"ep2d[dp]|DTI|MDDW|perf|PERFUSION|Gre Perf|BOLUS|MoCo", re.IGNORECASE)),
    # T2 FLAIR
    ("t2_flair", re.compile(r"FLAIR", re.IGNORECASE)),
    # T1 post-contrast
    ("t1_post", re.compile(r"T[1I].*(?:POST|stealth-post|GAD|\bC\b)", re.IGNORECASE)),
    # T2 (non-FLAIR)
    ("t2", re.compile(r"T2", re.IGNORECASE)),
    # T1 pre-contrast (general T1 without POST keyword)
    ("t1_pre", re.compile(r"T[1I]|t1", re.IGNORECASE)),
]


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


def classify_modality(series_desc: str) -> str | None:
    """Classify a series description into a structural modality or None (skip)."""
    for modality, pattern in MODALITY_RULES:
        if pattern.search(series_desc):
            return None if modality == "_skip" else modality
    return None


def collect_tasks_from_metadata(metadata_csv: Path, data_dir: Path, save_dir: Path) -> list[dict]:
    """
    Build conversion tasks from the TCIA download metadata CSV.
    """
    df = pd.read_csv(metadata_csv)
    required = {"Series Description", "Subject ID", "File Location"}
    if not required.issubset(df.columns):
        raise ValueError(f"Metadata CSV missing columns: {required - set(df.columns)}")

    mask = df["Series Description"].str.contains("ProcessedCaPTk", case=False, na=False)
    df_struct = df[mask].copy()

    df_struct["modality"] = df_struct["Series Description"].apply(classify_modality)
    df_struct = df_struct.dropna(subset=["modality"])

    df_struct = df_struct.sort_values(["Subject ID", "File Location"])

    # Resolve absolute paths and drop missing directories before assigning
    # study indices so that UIDs are contiguous for the data on disk.
    def _resolve(file_loc: str) -> Path | None:
        rel = file_loc[2:] if file_loc.startswith("./") else file_loc
        p = data_dir / rel
        return p if p.is_dir() else None

    df_struct["_series_dir"] = df_struct["File Location"].apply(_resolve)
    n_before = len(df_struct)
    df_struct = df_struct.dropna(subset=["_series_dir"])
    n_skipped = n_before - len(df_struct)
    if n_skipped:
        logger.warning(
            f"Skipped {n_skipped} series whose directories are not on disk "
            f"(incomplete download?)"
        )

    df_struct["study_idx"] = df_struct.groupby("Subject ID").cumcount()

    tasks = []
    for _, row in df_struct.iterrows():
        patient_id = row["Subject ID"]
        modality = row["modality"]
        study_idx = row["study_idx"]
        uid = f"{patient_id}_{study_idx}"

        tasks.append({
            "series_dir": str(row["_series_dir"]),
            "save_dir": str(save_dir),
            "uid": uid,
            "patient_id": patient_id,
            "modality": modality,
            "study_idx": study_idx,
            "series_desc": row["Series Description"],
        })
    return tasks


def collect_tasks_from_filesystem(dicom_root: Path, save_dir: Path) -> list[dict]:
    """Fallback: walk filesystem to discover DICOM series (when metadata CSV unavailable)."""
    tasks = []
    patient_dirs = sorted([d for d in dicom_root.iterdir() if d.is_dir()])

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        study_dirs = sorted([d for d in patient_dir.iterdir() if d.is_dir()])
        modality_counter = {}

        for study_dir in study_dirs:
            series_dirs = sorted([d for d in study_dir.iterdir() if d.is_dir()])
            for series_dir in series_dirs:
                # Extract series description from dir name: {num}-{desc}-{hash}
                parts = series_dir.name.split("-", 1)
                if len(parts) < 2:
                    continue
                desc_and_hash = parts[1]
                desc = desc_and_hash.rsplit("-", 1)[0] if "-" in desc_and_hash else desc_and_hash

                modality = classify_modality(desc)
                if modality is None:
                    continue

                dcm_files = list(series_dir.glob("*.dcm"))
                if not dcm_files:
                    continue

                idx = modality_counter.get(modality, 0)
                modality_counter[modality] = idx + 1
                uid = f"{patient_id}_{idx}"

                tasks.append({
                    "series_dir": str(series_dir),
                    "save_dir": str(save_dir),
                    "uid": uid,
                    "patient_id": patient_id,
                    "modality": modality,
                    "study_idx": idx,
                    "series_desc": desc,
                })
    return tasks


def convert_series_to_nifti(task: dict) -> dict | None:
    series_dir = Path(task["series_dir"])
    save_dir = Path(task["save_dir"])
    modality = task["modality"]
    uid = task["uid"]

    out_dir = save_dir / modality
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{uid}.nii.gz"

    if out_path.exists():
        logger.debug(f"Skipping (exists): {modality}/{out_path.name}")
        return None

    try:
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(series_dir))
        if not dicom_names:
            logger.warning(f"No DICOM series in {series_dir}")
            return None
        reader.SetFileNames(dicom_names)
        img = reader.Execute()
        sitk.WriteImage(img, str(out_path))

        # Extract DICOM metadata
        dcm_file = next(series_dir.glob("*.dcm"), None)
        md = {}
        if dcm_file:
            ds = pydicom.dcmread(str(dcm_file), stop_before_pixels=True)
            md = dataset2dict(ds)

        md.update({
            "nifti_path": str(out_path),
            "uid": uid,
            "patient_id": task["patient_id"],
            "modality": modality,
            "study_idx": task["study_idx"],
            "series_desc": task["series_desc"],
            "size": list(img.GetSize()),
            "spacing": list(img.GetSpacing()),
        })
        logger.info(f"Converted: {modality}/{out_path.name}")
        return md
    except Exception as e:
        logger.warning(f"Failed converting {series_dir}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Preprocess UPENN-GBM: structural DICOM to NIfTI")
    parser.add_argument(
        "--data-dir", type=str,
        default="/hpcwork/rwth1833/datasets/UPENN-GBM/UPENN-GBM_DownloadManifest20221129",
        help="Root folder containing UPENN-GBM download (with UPENN-GBM/ and metadata.csv)",
    )
    parser.add_argument(
        "--save-dir", type=str,
        default="/hpcwork/rwth1833/datasets/preprocessed/UPENN-GBM",
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

    metadata_csv = data_dir / "metadata.csv"

    if metadata_csv.exists():
        logger.info("Building tasks from metadata CSV ...")
        tasks = collect_tasks_from_metadata(metadata_csv, data_dir, save_dir)
    else:
        logger.info("Metadata CSV not found, scanning filesystem ...")
        dicom_root = data_dir / "UPENN-GBM"
        tasks = collect_tasks_from_filesystem(dicom_root, save_dir)

    logger.info(f"Found {len(tasks)} structural series to convert")
    modality_counts = {}
    for t in tasks:
        modality_counts[t["modality"]] = modality_counts.get(t["modality"], 0) + 1
    for m, c in sorted(modality_counts.items()):
        logger.info(f"  {m}: {c} series")

    logger.info("=" * 60)
    logger.info("Converting DICOM to NIfTI ...")
    logger.info("=" * 60)

    metadata_rows = []
    with Pool(processes=max(1, args.workers)) as pool:
        for md in tqdm(pool.imap_unordered(convert_series_to_nifti, tasks), total=len(tasks)):
            if md is not None:
                metadata_rows.append(md)

    df = pd.DataFrame(metadata_rows)
    df.to_csv(save_dir / "metadata.csv", index=False)

    total = sum(1 for _ in save_dir.rglob("*.nii.gz"))
    logger.info(f"Finished. NIfTI files written: {total}")
    logger.info("=" * 60)
    logger.info("Preprocessing completed.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
