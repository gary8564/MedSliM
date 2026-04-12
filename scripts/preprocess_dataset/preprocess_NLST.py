"""
Preprocess NLST (National Lung Screening Trial) dataset.
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


def collect_series_tasks(data_dir: Path, save_dir: Path) -> list[dict]:
    """Walk the NLST TCIA directory tree and collect one task per DICOM series."""
    tasks = []
    patient_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        study_dirs = sorted([d for d in patient_dir.iterdir() if d.is_dir()])

        for study_idx, study_dir in enumerate(study_dirs):
            series_dirs = sorted([d for d in study_dir.iterdir() if d.is_dir()])

            for series_dir in series_dirs:
                dcm_files = list(series_dir.glob('*.dcm'))
                if not dcm_files:
                    continue
                uid = f"{patient_id}_{study_idx}"
                tasks.append({
                    "series_dir": str(series_dir),
                    "save_dir": str(save_dir),
                    "uid": uid,
                    "patient_id": patient_id,
                    "study_idx": study_idx,
                    "study_dir_name": study_dir.name,
                    "series_dir_name": series_dir.name,
                })
    return tasks


def convert_series_to_nifti(task: dict) -> dict | None:
    series_dir = Path(task["series_dir"])
    save_dir = Path(task["save_dir"])
    uid = task["uid"]
    out_path = save_dir / f"{uid}.nii.gz"

    if out_path.exists():
        logger.debug(f"Skipping (exists): {out_path.name}")
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

        # Extract metadata from first DICOM
        dcm_file = next(series_dir.glob('*.dcm'), None)
        if dcm_file:
            ds = pydicom.dcmread(str(dcm_file), stop_before_pixels=True)
            md = dataset2dict(ds)
        else:
            md = {}

        md.update({
            "nifti_path": str(out_path),
            "uid": uid,
            "patient_id": task["patient_id"],
            "study_idx": task["study_idx"],
            "study_dir_name": task["study_dir_name"],
            "series_dir_name": task["series_dir_name"],
            "size": list(img.GetSize()),
            "spacing": list(img.GetSpacing()),
        })
        logger.info(f"Converted: {out_path.name}")
        return md
    except Exception as e:
        logger.warning(f"Failed converting {series_dir}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Preprocess NLST: DICOM to NIfTI")
    parser.add_argument(
        "--data-dir", type=str,
        default="/hpcwork/rwth1833/datasets/NLST/NLST-New-lesion-LongCT_source_series/NLST",
        help="Root folder containing per-patient DICOM directories",
    )
    parser.add_argument(
        "--save-dir", type=str,
        default="/hpcwork/rwth1833/datasets/preprocessed/NLST",
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
    tasks = collect_series_tasks(data_dir, save_dir)
    logger.info(f"Found {len(tasks)} series to convert")

    metadata_rows = []
    with Pool(processes=max(1, args.workers)) as pool:
        for md in tqdm(pool.imap_unordered(convert_series_to_nifti, tasks), total=len(tasks)):
            if md is not None:
                metadata_rows.append(md)

    df = pd.DataFrame(metadata_rows)
    df.to_csv(save_dir / "metadata.csv", index=False)

    num_files = len(list(save_dir.glob("*.nii.gz")))
    logger.info(f"Finished. NIfTI files written: {num_files}")
    logger.info("=" * 60)
    logger.info("Preprocessing completed.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
