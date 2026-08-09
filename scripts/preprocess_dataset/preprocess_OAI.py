"""
Preprocess OAI (Osteoarthritis Initiative) baseline MRI for MedSliM.

Converts series-level DICOM archives under image03/ to NIfTI:
  - SSL bucket (native diagnostic knee MRI)
      → {save_dir}/train/{sequence}/{plane}/{ID}.nii.gz
  - other bucket (thigh MRI only, for optional later use)
      → {save_dir}/other/{sequence}/{plane}/{ID}.nii.gz

Sequence folders are contrast/protocol only (plane prefix stripped), e.g.
  train/3d_dess/sagittal/, train/iw_tse/coronal/, other/t1_thigh/axial/
matching fastMRI's {split}/{sequence}/{plane}/ layout.

Converted sequences (whitelist)
  SSL:  SAG_3D_DESS_*, SAG_IW_TSE_*, COR_IW_TSE_*, COR_T1_3D_FLASH_*
  other: AX_T1_THIGH

Skipped (not converted) — rationale
  - MP_LOCATOR_*, PRESCRIPTION_*: multi-plane scouts / FOV setup; not analysis
    inputs. Stacking into one NIfTI is geometrically invalid.
  - COR_MPR_*, AX_MPR_*: reformats of the same SAG_3D_DESS acquisition
    (same pulse/contrast); redundant for SSL and inflate DESS content.
  - SAG_T2_MAP_*: multi-echo T2 mapping; SimpleITK would stack echoes into a
    fake volume. Only useful with per-echo split or a proper T2 fit.
  - All X-ray series and JPG thumbnails.

ID format (MedSliM multi-series grouping via _extract_exam_id):
  {src_subject_id}_{side}_MR{k}_{series_stem}
  e.g. 9363408_RIGHT_MR0_11304209
  MR{k} is assigned only among converted series within each exam.
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import shutil
import sys
import tarfile
import tempfile
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
import pydicom
import SimpleITK as sitk
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Native diagnostic knee sequences for MedSliM SSL (acquired, not derived).
SSL_SEQUENCE_PREFIXES: tuple[str, ...] = (
    "SAG_3D_DESS",
    "SAG_IW_TSE",
    "COR_IW_TSE",
    "COR_T1_3D_FLASH",
)

# Optional non-SSL keepers (thigh anatomy).
OTHER_SEQUENCE_PREFIXES: tuple[str, ...] = (
    "AX_T1_THIGH",
)

ESSENTIAL_COLUMNS = [
    "ID",
    "bucket",
    "split",
    "mri_sequence",
    "plane",
    "side",
    "src_subject_id",
    "exam_id",
    "image_description",
    "nifti_path",
    "tar_path",
    "series_stem",
    "mr_index",
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


def s3_to_local_rel(image_file: str) -> str | None:
    """Map NDA S3 path to relative path under image03/ (starts at 00m/)."""
    parts = image_file.strip().strip('"').split("/")
    try:
        j = parts.index("00m")
    except ValueError:
        return None
    return "/".join(parts[j:])


def canonicalize_sequence(image_description: str) -> str:
    """
    Map OAI image_description to the folder name:
    Strips side (LEFT/RIGHT/THIGH) and plane prefixes (SAG_/COR_/AX_), e.g.:
        SAG_3D_DESS_RIGHT  → 3d_dess   under train/3d_dess/sagittal/
        COR_IW_TSE_LEFT    → iw_tse    under train/iw_tse/coronal/
        AX_T1_THIGH        → t1_thigh  under other/t1_thigh/axial/
    Thigh is kept in the sequence name so it does not collide with knee T1.
    """
    desc = image_description.strip()
    is_thigh = "THIGH" in desc.upper()
    for side in ("_LEFT", "_RIGHT", "_THIGH"):
        if desc.endswith(side):
            desc = desc[: -len(side)]
            break
    desc = re.sub(r"\s+", "_", desc)
    # Strip plane prefix so sequence folders match {sequence}/{plane}.
    desc = re.sub(r"^(SAG|COR|AX)_", "", desc, flags=re.IGNORECASE)
    name = desc.lower()
    if is_thigh and "thigh" not in name:
        name = f"{name}_thigh"
    return name


def parse_side(image_description: str) -> str:
    desc = image_description.strip().upper()
    if desc.endswith("_LEFT") or desc.endswith(" LEFT"):
        return "LEFT"
    if desc.endswith("_RIGHT") or desc.endswith(" RIGHT"):
        return "RIGHT"
    if "THIGH" in desc:
        return "THIGH"
    return "UNKNOWN"


def parse_plane(image_description: str) -> str:
    desc = image_description.strip().upper()
    if desc.startswith("SAG") or "SAGITTAL" in desc:
        return "sagittal"
    if desc.startswith("COR") or "CORONAL" in desc:
        return "coronal"
    if desc.startswith("AX") or "AXIAL" in desc:
        return "axial"
    return "axial"


def is_ssl_sequence(image_description: str) -> bool:
    desc = image_description.strip().upper()
    return any(desc.startswith(p) for p in SSL_SEQUENCE_PREFIXES)


def is_other_sequence(image_description: str) -> bool:
    desc = image_description.strip().upper()
    return any(desc.startswith(p) for p in OTHER_SEQUENCE_PREFIXES)


def route_bucket(image_description: str) -> str | None:
    """Return 'ssl', 'other', or None if the series should not be converted."""
    if is_ssl_sequence(image_description):
        return "ssl"
    if is_other_sequence(image_description):
        return "other"
    return None


def load_image03_rows(manifest_path: Path) -> list[dict]:
    """Parse NDA image03.txt (two header rows) into dicts."""
    rows: list[dict] = []
    with open(manifest_path, newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        next(reader)  # definition row
        colnames = [h.strip().strip('"') for h in header]
        for raw in reader:
            if not raw or raw[0].startswith("collection"):
                continue
            row = {
                colnames[i]: raw[i].strip().strip('"') if i < len(raw) else ""
                for i in range(len(colnames))
            }
            rows.append(row)
    return rows


def collect_tasks(
    data_dir: Path,
    save_dir: Path,
    *,
    bucket_filter: str = "all",
) -> list[dict]:
    """Build conversion tasks from image03.txt (whitelisted MRI only)."""
    manifest = data_dir / "image03.txt"
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    raw_rows = load_image03_rows(manifest)
    # Index only convertible series so MR{k} is stable across --bucket filters.
    exam_series: dict[str, list[dict]] = defaultdict(list)
    skipped_non_mri = 0
    skipped_bad_path = 0
    skipped_sequence = 0

    convertible: list[dict] = []
    for row in raw_rows:
        if row.get("image_modality", "").upper() != "MRI":
            skipped_non_mri += 1
            continue
        desc = row.get("image_description", "")
        bucket = route_bucket(desc)
        if bucket is None:
            skipped_sequence += 1
            continue
        rel = s3_to_local_rel(row.get("image_file", ""))
        if rel is None:
            skipped_bad_path += 1
            continue
        # Defer existence checks to the worker (avoids many Lustre stats at startup).
        tar_path = data_dir / "image03" / rel

        side = parse_side(desc)
        subject = row.get("src_subject_id", "")
        series_stem = tar_path.name.removesuffix(".tar.gz")
        exam_key = f"{subject}_{side}"
        meta = {
            "src_subject_id": subject,
            "side": side,
            "exam_key": exam_key,
            "image_description": desc,
            "mri_sequence": canonicalize_sequence(desc),
            "plane": parse_plane(desc),
            "bucket": bucket,
            "tar_path": str(tar_path),
            "series_stem": series_stem,
            "subjectkey": row.get("subjectkey", ""),
            "interview_date": row.get("interview_date", ""),
            "sex": row.get("sex", ""),
            "visit": row.get("visit", ""),
            "study_release": row.get("study", ""),
        }
        convertible.append(meta)
        exam_series[exam_key].append(meta)

    # Stable MR index within each exam (sorted by series_stem for determinism).
    for exam_key, series_list in exam_series.items():
        series_list.sort(key=lambda t: t["series_stem"])
        for k, t in enumerate(series_list):
            t["mr_index"] = k
            t["ID"] = f"{exam_key}_MR{k}_{t['series_stem']}"
            t["exam_id"] = exam_key
            t["save_dir"] = str(save_dir)
            t["force"] = False

    if bucket_filter == "all":
        candidates = convertible
    else:
        candidates = [t for t in convertible if t["bucket"] == bucket_filter]

    tasks = sorted(
        candidates,
        key=lambda t: (t["bucket"], t["src_subject_id"], t["side"], t["mr_index"]),
    )
    logger.info(
        f"Collected {len(tasks)} convertible MRI series "
        f"(skipped non-MRI={skipped_non_mri}, sequence={skipped_sequence}, "
        f"bad_path={skipped_bad_path})"
    )
    n_ssl = sum(1 for t in convertible if t["bucket"] == "ssl")
    n_other = sum(1 for t in convertible if t["bucket"] == "other")
    logger.info(
        f"  Convertible: {len(convertible)} (SSL={n_ssl}, other={n_other}); "
        f"converting bucket={bucket_filter!r}: {len(tasks)}"
    )
    return tasks


def _list_dicom_files(dicom_dir: Path) -> list[str]:
    """List DICOM-like files (including extensionless OAI members)."""
    files = sorted(
        str(p)
        for p in dicom_dir.rglob("*")
        if p.is_file() and not p.name.startswith(".")
    )
    return files


def _extract_tar_to_temp(tar_path: Path) -> Path:
    """Extract a series tar.gz into a unique temporary directory."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="oai_dcm_"))
    with tarfile.open(tar_path, "r:gz") as tf:
        if hasattr(tarfile, "data_filter"):
            tf.extractall(tmp_dir, filter="data")
        else:
            tf.extractall(tmp_dir)
    return tmp_dir


def convert_series_to_nifti(task: dict) -> dict | None:
    """Extract one OAI DICOM tar.gz and convert to NIfTI."""
    tar_path = Path(task["tar_path"])
    save_dir = Path(task["save_dir"])
    bucket = task["bucket"]
    sequence = task["mri_sequence"]
    plane = task["plane"]
    uid = task["ID"]
    force = bool(task.get("force", False))

    if bucket == "ssl":
        out_dir = save_dir / "train" / sequence / plane
        split = "train"
    else:
        out_dir = save_dir / "other" / sequence / plane
        split = "other"

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{uid}.nii.gz"

    base_md = {
        "ID": uid,
        "bucket": bucket,
        "split": split,
        "mri_sequence": sequence,
        "plane": plane,
        "side": task["side"],
        "src_subject_id": task["src_subject_id"],
        "exam_id": task["exam_id"],
        "image_description": task["image_description"],
        "nifti_path": str(out_file),
        "tar_path": str(tar_path),
        "series_stem": task["series_stem"],
        "mr_index": task["mr_index"],
        "subjectkey": task.get("subjectkey", ""),
        "interview_date": task.get("interview_date", ""),
        "sex": task.get("sex", ""),
        "visit": task.get("visit", ""),
        "study_release": task.get("study_release", ""),
    }

    if out_file.exists() and not force:
        logger.debug(f"Skipping (exists): {out_file}")
        return base_md

    if not tar_path.is_file():
        logger.warning(f"Missing archive: {tar_path}")
        return None

    tmp_dir: Path | None = None
    try:
        tmp_dir = _extract_tar_to_temp(tar_path)

        # Prefer leaf directory that actually holds the slices.
        dicom_dirs = [p for p in tmp_dir.rglob("*") if p.is_dir()]
        candidate_dirs = [tmp_dir] + dicom_dirs

        reader = sitk.ImageSeriesReader()
        reader.SetSpacingWarningRelThreshold(0.01)
        dicom_names: list[str] = []
        used_dir: Path | None = None
        for d in candidate_dirs:
            names = list(reader.GetGDCMSeriesFileNames(str(d)))
            if names:
                dicom_names = names
                used_dir = d
                break

        if not dicom_names:
            # Extensionless DICOM fallback: feed sorted files directly.
            for d in candidate_dirs:
                files = _list_dicom_files(d)
                if files:
                    dicom_names = files
                    used_dir = d
                    break

        if not dicom_names:
            raise RuntimeError(f"No DICOM files found after extracting {tar_path}")

        reader.SetFileNames(dicom_names)
        img = reader.Execute()
        sitk.WriteImage(img, str(out_file), useCompression=True, compressionLevel=1)

        try:
            ds = read_dicom_file(used_dir if used_dir is not None else tmp_dir)
            md = dataset2dict(ds)
            md.update(base_md)
        except Exception:
            md = dict(base_md)
        return md
    except Exception as e:
        logger.warning(f"Failed conversion for {tar_path}: {e}")
        if out_file.exists():
            try:
                out_file.unlink()
            except OSError:
                pass
        return None
    finally:
        if tmp_dir is not None and tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


def filter_essential(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ESSENTIAL_COLUMNS if c in df.columns]
    return df[cols]


def _merge_csv_by_id(path: Path, df_new: pd.DataFrame) -> pd.DataFrame:
    """Merge new rows into an existing CSV keyed by ID (new rows win)."""
    if df_new.empty and path.exists():
        return pd.read_csv(path, dtype={"ID": str})
    if path.exists() and path.stat().st_size > 0:
        df_old = pd.read_csv(path, dtype={"ID": str})
        if not df_old.empty and "ID" in df_old.columns and "ID" in df_new.columns:
            df_old = df_old[~df_old["ID"].isin(df_new["ID"])]
            df = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df = df_new
    else:
        df = df_new
    if "ID" in df.columns:
        df = df.sort_values("ID").reset_index(drop=True)
    return df


def copy_enrollee(data_dir: Path, save_dir: Path) -> None:
    src = data_dir / "oai_enrollee01.txt"
    if not src.exists():
        logger.warning(f"Enrollee file not found: {src}")
        return
    dst = save_dir / "oai_enrollee01.txt"
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        shutil.copy2(src, dst)
        logger.info(f"Copied {src.name} -> {dst}")
    else:
        logger.debug(f"Enrollee already present: {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess OAI baseline MRI: DICOM tar.gz → NIfTI "
            "(SSL: DESS/IW-TSE/FLASH; other: thigh T1; skips locators/MPR/T2_MAP)"
        )
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/hpcwork/rwth1833/datasets/OAI",
        help="Root folder of the downloaded OAI NDA package",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="/hpcwork/rwth1833/datasets/preprocessed/OAI",
        help="Output directory for NIfTI files and metadata CSVs",
    )
    parser.add_argument(
        "--bucket",
        type=str,
        choices=["ssl", "other", "all"],
        default="all",
        help="Which sequence bucket to convert (default: all)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Convert only the first N tasks (smoke test)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reconvert even if output NIfTI already exists",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    num_workers = args.workers or os.cpu_count() or 8

    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Step 1: Collecting MRI series from image03.txt")
    logger.info("=" * 60)
    tasks = collect_tasks(data_dir, save_dir, bucket_filter=args.bucket)
    for t in tasks:
        t["force"] = args.force

    if args.limit is not None:
        tasks = tasks[: max(0, int(args.limit))]
        logger.info(f"Limited to first {len(tasks)} tasks (--limit)")

    if not tasks:
        logger.error("No tasks to convert.")
        return

    logger.info("=" * 60)
    logger.info(f"Step 2: Converting DICOM → NIfTI with {num_workers} workers")
    logger.info("=" * 60)
    metadata_rows: list[dict] = []
    with Pool(processes=num_workers, maxtasksperchild=50) as pool:
        chunksize = max(1, len(tasks) // (num_workers * 4))
        for md in tqdm(
            pool.imap_unordered(convert_series_to_nifti, tasks, chunksize=chunksize),
            total=len(tasks),
        ):
            if md is not None:
                metadata_rows.append(md)

    logger.info("=" * 60)
    logger.info("Step 3: Writing metadata CSVs")
    logger.info("=" * 60)
    df_meta = pd.DataFrame(metadata_rows)
    if df_meta.empty:
        logger.error("No successful conversions; not writing CSVs.")
        return

    # Merge so partial --bucket / --limit runs do not wipe prior CSVs.
    df_meta = _merge_csv_by_id(save_dir / "metadata.csv", df_meta)
    df_meta.to_csv(save_dir / "metadata.csv", index=False)
    logger.info(f"Wrote metadata.csv ({len(df_meta)} rows)")

    df_ssl_new = filter_essential(df_meta[df_meta["bucket"] == "ssl"].copy())
    df_other_new = filter_essential(df_meta[df_meta["bucket"] == "other"].copy())
    # Re-filter from merged metadata so train/other stay consistent with metadata.csv
    df_ssl = df_ssl_new
    df_other = df_other_new
    df_ssl.to_csv(save_dir / "train.csv", index=False)
    df_other.to_csv(save_dir / "other.csv", index=False)
    logger.info(f"Wrote train.csv ({len(df_ssl)} SSL series)")
    logger.info(f"Wrote other.csv ({len(df_other)} other series)")

    copy_enrollee(data_dir, save_dir)

    n_nifti = len(list(save_dir.rglob("*.nii.gz")))
    logger.info(f"Finished. NIfTI files on disk: {n_nifti}")
    logger.info("=" * 60)
    logger.info("Preprocessing completed.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
