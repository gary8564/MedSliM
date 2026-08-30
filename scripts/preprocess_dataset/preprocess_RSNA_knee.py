"""
Preprocess the RSNA Knee Abnormality Detection dataset for MedSliM.

The Kaggle dataset contains several DICOM series per knee MRI study. Series
metadata supplies the anatomical plane and two sequence characteristics:
fluid sensitivity and fat suppression.

Input layout:
  {data_dir}/{train,test}.csv
  {data_dir}/{train,test}_series.csv
  {data_dir}/{train,test}_series/{StudyInstanceUID}/{SeriesInstanceUID}/*.dcm

Output layout:
  {save_dir}/{split}/{mri_sequence}/{plane}/{ID}.nii.gz
  {save_dir}/{train,test}.csv
  {save_dir}/metadata.csv

IDs use ``{StudyInstanceUID}_MR{k}_{SeriesInstanceUID}``. MedSliM therefore
groups all series from one study into the same exam while retaining globally
unique series IDs.

Reference:
https://www.kaggle.com/competitions/rsna-knee-abnormality-detection
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import pydicom
import SimpleITK as sitk
from tqdm import tqdm


logger = logging.getLogger(__name__)

SERIES_COLUMNS = [
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "Fluid_Sensitive",
    "Fat_Suppression",
    "Anatomical_Plane",
]

DICOM_COLUMNS = [
    "MagneticFieldStrength",
    "Manufacturer",
    "ManufacturerModelName",
    "RepetitionTime",
    "EchoTime",
    "FlipAngle",
    "MRAcquisitionType",
    "SliceThickness",
    "SpacingBetweenSlices",
    "PixelSpacing",
    "Rows",
    "Columns",
    "SeriesDescription",
    "ProtocolName",
    "SequenceName",
]

BASE_OUTPUT_COLUMNS = [
    "ID",
    "split",
    "mri_sequence",
    "plane",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "mr_index",
    "Fluid_Sensitive",
    "Fat_Suppression",
    "nifti_path",
    "dicom_dir",
    "size",
    "spacing",
]


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)


def maybe_convert(value):
    """Convert common pydicom values into CSV-serializable Python values."""
    if isinstance(value, (pydicom.sequence.Sequence, pydicom.dataset.Dataset)):
        return None
    if isinstance(value, pydicom.multival.MultiValue):
        return list(value)
    if isinstance(value, pydicom.valuerep.PersonName):
        return str(value)
    return value


def selected_dicom_metadata(ds: pydicom.dataset.Dataset) -> dict:
    metadata = {}
    for keyword in DICOM_COLUMNS:
        if keyword not in ds:
            continue
        value = maybe_convert(ds.data_element(keyword).value)
        if value is not None:
            metadata[keyword] = value
    return metadata


def sequence_name(fluid_sensitive: int, fat_suppression: int) -> str:
    """Create a stable sequence bucket from the competition annotations."""
    fluid = bool(int(fluid_sensitive))
    fat_sat = bool(int(fat_suppression))
    base = "fluid_sensitive" if fluid else "non_fluid_sensitive"
    return f"{base}_fs" if fat_sat else base


def normalize_plane(plane: str) -> str:
    normalized = str(plane).strip().lower()
    if normalized not in {"axial", "coronal", "sagittal"}:
        raise ValueError(f"Unsupported anatomical plane: {plane!r}")
    return normalized


def _read_split_tables(data_dir: Path, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    studies_path = data_dir / f"{split}.csv"
    series_path = data_dir / f"{split}_series.csv"
    if not studies_path.is_file():
        raise FileNotFoundError(f"Study CSV not found: {studies_path}")
    if not series_path.is_file():
        raise FileNotFoundError(f"Series CSV not found: {series_path}")

    studies = pd.read_csv(studies_path, dtype={"StudyInstanceUID": str})
    series = pd.read_csv(
        series_path,
        dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str},
    )
    missing = [column for column in SERIES_COLUMNS if column not in series.columns]
    if missing:
        raise ValueError(f"{series_path} is missing columns: {missing}")
    if "StudyInstanceUID" not in studies.columns:
        raise ValueError(f"{studies_path} is missing StudyInstanceUID")
    return studies, series


def collect_tasks(
    data_dir: Path,
    save_dir: Path,
    splits: list[str],
    *,
    force: bool = False,
) -> tuple[list[dict], dict[str, list[str]]]:
    tasks: list[dict] = []
    study_columns_by_split: dict[str, list[str]] = {}

    for split in splits:
        studies, series = _read_split_tables(data_dir, split)
        study_columns_by_split[split] = [
            column for column in studies.columns if column != "StudyInstanceUID"
        ]
        merged = series.merge(
            studies,
            on="StudyInstanceUID",
            how="left",
            validate="many_to_one",
        )
        merged = merged.sort_values(
            ["StudyInstanceUID", "SeriesInstanceUID"], kind="stable"
        ).reset_index(drop=True)
        merged["mr_index"] = merged.groupby("StudyInstanceUID").cumcount()

        for row in merged.to_dict(orient="records"):
            study_uid = str(row["StudyInstanceUID"])
            series_uid = str(row["SeriesInstanceUID"])
            mr_index = int(row["mr_index"])
            dicom_dir = (
                data_dir / f"{split}_series" / study_uid / series_uid
            )
            task = dict(row)
            task.update(
                {
                    "ID": f"{study_uid}_MR{mr_index}_{series_uid}",
                    "split": split,
                    "dicom_dir": str(dicom_dir),
                    "save_dir": str(save_dir),
                    "mri_sequence": sequence_name(
                        row["Fluid_Sensitive"], row["Fat_Suppression"]
                    ),
                    "plane": normalize_plane(row["Anatomical_Plane"]),
                    "force": force,
                }
            )
            tasks.append(task)

    return tasks, study_columns_by_split


def _first_dicom(dicom_dir: Path) -> Path:
    dicom_path = next(dicom_dir.glob("*.dcm"), None)
    if dicom_path is None:
        dicom_path = next((path for path in dicom_dir.iterdir() if path.is_file()), None)
    if dicom_path is None:
        raise FileNotFoundError(f"No DICOM files found in {dicom_dir}")
    return dicom_path


def _base_metadata(task: dict, out_path: Path, image: sitk.Image) -> dict:
    excluded = {"save_dir", "force", "Anatomical_Plane"}
    metadata = {key: value for key, value in task.items() if key not in excluded}
    metadata.update(
        {
            "nifti_path": str(out_path),
            "size": list(image.GetSize()),
            "spacing": list(image.GetSpacing()),
        }
    )
    return metadata


def convert_series_to_nifti(task: dict) -> dict | None:
    dicom_dir = Path(task["dicom_dir"])
    save_dir = Path(task["save_dir"])
    out_path = (
        save_dir
        / task["split"]
        / task["mri_sequence"]
        / task["plane"]
        / f"{task['ID']}.nii.gz"
    )

    try:
        if not dicom_dir.is_dir():
            raise FileNotFoundError(f"Series directory not found: {dicom_dir}")

        first_dicom = _first_dicom(dicom_dir)
        ds = pydicom.dcmread(str(first_dicom), stop_before_pixels=True)

        if out_path.exists() and not task["force"]:
            image = sitk.ReadImage(str(out_path))
        else:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            reader = sitk.ImageSeriesReader()
            reader.SetSpacingWarningRelThreshold(0.01)
            filenames = list(
                reader.GetGDCMSeriesFileNames(
                    str(dicom_dir), str(task["SeriesInstanceUID"])
                )
            )
            if not filenames:
                filenames = list(reader.GetGDCMSeriesFileNames(str(dicom_dir)))
            if not filenames:
                raise RuntimeError("GDCM did not discover a DICOM series")

            reader.SetFileNames(filenames)
            image = reader.Execute()
            sitk.WriteImage(
                image,
                str(out_path),
                useCompression=True,
                compressionLevel=1,
            )

        metadata = selected_dicom_metadata(ds)
        metadata.update(_base_metadata(task, out_path, image))
        return metadata
    except Exception as exc:
        logger.warning(
            "Failed conversion for %s/%s: %s",
            task.get("StudyInstanceUID", "unknown"),
            task.get("SeriesInstanceUID", "unknown"),
            exc,
        )
        if task.get("force") and out_path.exists():
            try:
                out_path.unlink()
            except OSError:
                pass
        return None


def merge_metadata(path: Path, new_rows: pd.DataFrame) -> pd.DataFrame:
    """Merge partial/resumed runs by ID, with new rows taking precedence."""
    if path.exists() and path.stat().st_size:
        old_rows = pd.read_csv(path, dtype={"ID": str})
        if not new_rows.empty:
            old_rows = old_rows[~old_rows["ID"].isin(new_rows["ID"])]
            merged = pd.concat([old_rows, new_rows], ignore_index=True)
        else:
            merged = old_rows
    else:
        merged = new_rows

    if "ID" in merged.columns:
        merged = merged.sort_values(["split", "ID"], kind="stable").reset_index(drop=True)
    return merged


def split_output_columns(
    df: pd.DataFrame,
    study_columns: list[str],
) -> list[str]:
    requested = BASE_OUTPUT_COLUMNS + study_columns + DICOM_COLUMNS
    return [column for column in requested if column in df.columns]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess RSNA Knee Abnormality Detection DICOM series to NIfTI"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/hpcwork/rwth1833/datasets/RSNA-Knee",
        help="Root containing train/test CSV files and *_series directories",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="/hpcwork/rwth1833/datasets/preprocessed/RSNA-Knee",
        help="Output directory for NIfTI volumes and metadata CSV files",
    )
    parser.add_argument(
        "--split",
        choices=["train", "test", "all"],
        default="all",
        help="Dataset split to convert (default: all)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Parallel workers (default: available CPU count)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Convert only the first N series, useful for a smoke test",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reconvert NIfTI files that already exist",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)

    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {data_dir}")
    save_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "test"] if args.split == "all" else [args.split]
    workers = max(1, args.workers or os.cpu_count() or 8)

    logger.info("=" * 60)
    logger.info("Step 1: Collecting series tasks")
    logger.info("=" * 60)
    tasks, study_columns = collect_tasks(
        data_dir, save_dir, splits, force=args.force
    )
    logger.info("Found %d DICOM series", len(tasks))
    for split in splits:
        count = sum(task["split"] == split for task in tasks)
        logger.info("  %s: %d series", split, count)

    if args.limit is not None:
        tasks = tasks[: max(0, int(args.limit))]
        logger.info("Limited conversion to %d series", len(tasks))
    if not tasks:
        logger.warning("No conversion tasks selected")
        return

    logger.info("=" * 60)
    logger.info("Step 2: Converting DICOM to NIfTI with %d workers", workers)
    logger.info("=" * 60)
    metadata_rows: list[dict] = []
    chunksize = max(1, len(tasks) // (workers * 4))
    with Pool(processes=workers, maxtasksperchild=50) as pool:
        for metadata in tqdm(
            pool.imap_unordered(
                convert_series_to_nifti, tasks, chunksize=chunksize
            ),
            total=len(tasks),
        ):
            if metadata is not None:
                metadata_rows.append(metadata)

    logger.info("=" * 60)
    logger.info("Step 3: Writing metadata CSV files")
    logger.info("=" * 60)
    new_metadata = pd.DataFrame(metadata_rows)
    metadata_path = save_dir / "metadata.csv"
    metadata = merge_metadata(metadata_path, new_metadata)
    if metadata.empty:
        logger.error("No successful conversions; no metadata was written")
        return
    metadata.to_csv(metadata_path, index=False)
    logger.info("Wrote metadata.csv with %d rows", len(metadata))

    for split in splits:
        split_metadata = metadata[metadata["split"] == split].copy()
        columns = split_output_columns(
            split_metadata, study_columns.get(split, [])
        )
        split_metadata[columns].to_csv(save_dir / f"{split}.csv", index=False)
        logger.info("Wrote %s.csv with %d rows", split, len(split_metadata))

    nifti_count = sum(1 for _ in save_dir.rglob("*.nii.gz"))
    logger.info("NIfTI files on disk: %d", nifti_count)
    logger.info(
        "Successful in this run: %d/%d", len(metadata_rows), len(tasks)
    )
    logger.info("=" * 60)
    logger.info("Preprocessing completed")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
