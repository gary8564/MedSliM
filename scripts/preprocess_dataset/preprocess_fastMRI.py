import argparse
import logging
import os
import re
import sys
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from tqdm import tqdm


logger = logging.getLogger(__name__)

# Essential columns to keep in the output metadata CSV
# These are the most useful for MRI analysis and downstream tasks
ESSENTIAL_COLUMNS = [
    # Identifiers
    "ID",
    "split",
    "mri_sequence",
    "plane",
    "nifti_path",
    "dicom_dir",
    "exam_name",
    "study_name",
    "series_name",
    "batch_name",
    # Acquisition parameters
    "MagneticFieldStrength",
    "Manufacturer",
    "ManufacturerModelName",
    "RepetitionTime",
    "EchoTime",
    "FlipAngle",
    "MRAcquisitionType",
    # Geometry
    "SliceThickness",
    "SpacingBetweenSlices",
    "PixelSpacing",
    "Rows",
    "Columns",
    # Series info
    "SeriesDescription",
    "ProtocolName",
    # Counts
    "NumberOfFrames",
    "ImagesInAcquisition",
]


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


def filter_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Filter metadata DataFrame to keep only essential columns."""
    cols_to_keep = [c for c in ESSENTIAL_COLUMNS if c in df.columns]
    return df[cols_to_keep]


def maybe_convert(x):
    if isinstance(x, pydicom.sequence.Sequence):
        return None
    if isinstance(x, pydicom.dataset.Dataset):
        return None
    if isinstance(x, pydicom.multival.MultiValue):
        return list(x)
    if isinstance(x, pydicom.valuerep.PersonName):
        return str(x)
    return x


def dataset2dict(ds, exclude=("PixelData", "")):
    result = {}
    for key in ds.keys():
        kw = ds[key].keyword
        if kw in exclude:
            continue
        val = maybe_convert(ds[key].value)
        if val is not None:
            result[kw] = val
    return result


def read_dicom_file(dicom_dir: Path):
    dicom_file = next(dicom_dir.glob("*.dcm"), None)
    if dicom_file is None:
        dicom_file = next((p for p in dicom_dir.iterdir() if p.is_file()), None)
    if dicom_file is None:
        raise FileNotFoundError(f"No DICOM files found in {dicom_dir}")
    return pydicom.dcmread(str(dicom_file), stop_before_pixels=True)


def detect_plane(ds) -> str:
    text = " ".join(
        t
        for t in [
            getattr(ds, "SeriesDescription", ""),
            getattr(ds, "ProtocolName", ""),
            getattr(ds, "SequenceName", ""),
        ]
        if t
    ).lower()
    if "sag" in text:
        return "sagittal"
    if "cor" in text:
        return "coronal"
    if "ax" in text:
        return "axial"

    iop = getattr(ds, "ImageOrientationPatient", None)
    if iop and len(iop) >= 6:
        row = np.array(iop[:3], dtype=float)
        col = np.array(iop[3:6], dtype=float)
        normal = np.cross(row, col)
        absn = np.abs(normal)
        if absn[0] >= absn[1] and absn[0] >= absn[2]:
            return "sagittal"
        if absn[1] >= absn[0] and absn[1] >= absn[2]:
            return "coronal"
        if absn[2] >= absn[0] and absn[2] >= absn[1]:
            return "axial"
    else:
        raise ValueError(f"Couldn't detect plane from DICOM metadata.")

def detect_sequence(ds) -> str:
    text = " ".join(
        t
        for t in [
            getattr(ds, "SeriesDescription", ""),
            getattr(ds, "ProtocolName", ""),
            getattr(ds, "SequenceName", ""),
        ]
        if t
    ).lower()

    if "t2" in text:
        base = "t2"
    elif "pd" in text or "proton density" in text:
        base = "pd"
    else:
        raise ValueError(f"Couldn't detect sequence from DICOM metadata.")

    fat_suppressed = re.search(
        r"(fat\s*sat|fat[-_ ]?supp|fatsat|fs\b|_fs|fs_)",
        text,
    )
    if fat_suppressed:
        return f"{base}_fs"
    return base


def find_batch_dirs(data_root: Path) -> list[Path]:
    batch_dirs = [
        p
        for p in data_root.iterdir()
        if p.is_dir() and p.name.startswith("knee_mri_clinical_seq")
    ]
    return batch_dirs if batch_dirs else [data_root]


def collect_series_tasks(data_root: Path) -> list[dict]:
    tasks = []
    batch_dirs = find_batch_dirs(data_root)
    for batch_dir in batch_dirs:
        for exam_dir in batch_dir.iterdir():
            if not exam_dir.is_dir():
                continue
            for study_dir in exam_dir.iterdir():
                if not (study_dir.is_dir() and study_dir.name.startswith("study_")):
                    continue
                for series_dir in study_dir.iterdir():
                    if not (series_dir.is_dir() and series_dir.name.startswith("MR")):
                        continue
                    tasks.append(
                        {
                            "dicom_dir": str(series_dir),
                            "exam_name": exam_dir.name,
                            "study_name": study_dir.name,
                            "series_name": series_dir.name,
                            "batch_name": batch_dir.name,
                        }
                    )
    tasks = sorted(
        tasks,
        key=lambda t: (t["exam_name"], t["study_name"], t["series_name"], t["batch_name"]),
    )
    return tasks

def convert_series_to_nifti(task: dict) -> dict | None:
    """Convert DICOM series to NIfTI"""
    dicom_dir = Path(task["dicom_dir"])
    save_dir = Path(task["save_dir"])

    try:
        # Read DICOM metadata to determine output path
        ds = read_dicom_file(dicom_dir)
        plane = detect_plane(ds)
        sequence = detect_sequence(ds)

        # Use study series as UID (e.g., "study_ee50e35b_MR4_0b2ac15c")
        uid = f"{task['study_name']}_{task['series_name']}"

        out_dir = save_dir / task["split"] / sequence / plane
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{uid}.nii.gz"
        
        # Skip if already converted (resume capability)
        if out_file.exists():
            md = dataset2dict(ds)
            md.update({
                "ID": uid,
                "split": task["split"],
                "mri_sequence": sequence,
                "plane": plane,
                "nifti_path": str(out_file),
                "dicom_dir": str(dicom_dir),
                "exam_name": task["exam_name"],
                "study_name": task["study_name"],
                "series_name": task["series_name"],
                "batch_name": task["batch_name"],
            })
            return md

        reader = sitk.ImageSeriesReader()
        reader.SetSpacingWarningRelThreshold(0.01)
        dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_dir))
        if not dicom_names:
            raise RuntimeError("No series files discovered via GDCM in directory")
        
        reader.SetFileNames(dicom_names)
        img = reader.Execute()
        sitk.WriteImage(img, str(out_file), useCompression=True, compressionLevel=1)

        md = dataset2dict(ds)
        md.update(
            {
                "ID": uid,
                "split": task["split"],
                "mri_sequence": sequence,
                "plane": plane,
                "nifti_path": str(out_file),
                "dicom_dir": str(dicom_dir),
                "exam_name": task["exam_name"],
                "study_name": task["study_name"],
                "series_name": task["series_name"],
                "batch_name": task["batch_name"],
            }
        )
        return md
    except Exception as e:
        logger.warning(f"Failed conversion for {dicom_dir}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Preprocess fastMRI knee DICOM to NIfTI")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/hpcwork/rwth1833/datasets/fastMRI/knee",
        help="Root folder containing fastMRI knee DICOM data",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="/work/rwth1833/datasets/preprocessed/fastMRI",
        help="Output directory for NIfTI and metadata",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="train",
        help="Split name used in output folder structure (train/val/test)",
    )
    parser.add_argument("--workers", type=int, default=None, 
                        help="Number of parallel workers (default: number of CPUs)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    
    # Use all available CPUs by default
    num_workers = args.workers or os.cpu_count() or 8

    data_root = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("================================================")
    logger.info(f"Step 1: Converting DICOM to NIfTI with workers: {num_workers} ...")
    logger.info("================================================")
    tasks = collect_series_tasks(data_root)
    logger.info(f"Found {len(tasks)} series folders to convert")
    for task in tasks:
        task["split"] = args.split
        task["save_dir"] = str(save_dir)
    
    metadata_rows = []
    with Pool(processes=num_workers, maxtasksperchild=100) as pool:
        chunksize = max(1, len(tasks) // (num_workers * 4))
        for md in tqdm(pool.imap_unordered(convert_series_to_nifti, tasks, chunksize=chunksize), total=len(tasks)):
            if md is not None:
                metadata_rows.append(md)

    df_meta = pd.DataFrame(metadata_rows)
    df_meta.to_csv(save_dir / "metadata.csv", index=False)
    
    logger.info("================================================")
    logger.info("Step 2: Filtering metadata to keep only essential columns")
    logger.info("================================================")
    logger.info(f"Loading metadata from {save_dir / 'metadata.csv'} ...")
    df_meta = pd.read_csv(save_dir / "metadata.csv")
    df_meta_filtered = filter_metadata(df_meta)
    df_meta_filtered.to_csv(save_dir / f"{args.split}.csv", index=False)
    logger.info(f"Saved filtered metadata to {args.split}.csv.")

    num_series = len(list(save_dir.rglob("*.nii.gz")))
    logger.info(f"Finished. NIfTI files written: {num_series}")

    logger.info("================================================")
    logger.info("Preprocessing completed.")
    logger.info("================================================")


if __name__ == "__main__":
    main()
