"""
Preprocess the SKM-TEA (Stanford Knee MRI Multi-Task Evaluation) dataset.

Converts DICOM images to NIfTI volumes (one per echo) and extracts multilabel
classification labels from the COCO-style annotation JSONs.

SKM-TEA provides quantitative DESS (qDESS) knee MRI with 2 echoes:
    Echo 1 (short TE ≈ 6 ms, T1-weighted-like)  → DESS_E1/{scan_id}.nii.gz
    Echo 2 (long  TE ≈ 34 ms, T2-weighted-like) → DESS_E2/{scan_id}.nii.gz

Echo 1 provides strong anatomical contrast; Echo 2 is more sensitive to
fluid and edema (useful for effusions, ligamentous injuries).

Input data structure:
    {data_dir}/dicoms/{scan_id}.tar.gz (raw downloaded tarball)
    {data_dir}/raw_images/{scan_id}/*.dcm
    {data_dir}/annotations/v1.0.0/{train,val,test}.json

Preprocessed output data structure:
    {save_dir}/{split}/sagittal/DESS_E1/{scan_id}.nii.gz   (Echo 1)
    {save_dir}/{split}/sagittal/DESS_E2/{scan_id}.nii.gz   (Echo 2)
    {save_dir}/{split}.csv

Classification labels:
    meniscal_tear
    ligament_tear
    cartilage_lesion
    effusion
"""
import argparse
import json
import logging
import sys
import tarfile
import warnings
from pathlib import Path
from multiprocessing import Pool
from typing import Sequence, Union

import numpy as np
import pandas as pd
import pydicom
import torchio as tio
from tqdm import tqdm

from med_slim.utils.preprocessing.slice_axis_resolver import build_slice_last_affine

logger = logging.getLogger(__name__)

SUPERCATEGORY_COLUMNS = {
    "Meniscal Tear": "meniscal_tear",
    "Ligament Tear": "ligament_tear",
    "Cartilage Lesion": "cartilage_lesion",
    "Effusion": "effusion",
}
LABEL_COLUMNS = list(SUPERCATEGORY_COLUMNS.values())

# Echo number → subdirectory name (sequence type)
ECHO_TO_SEQ = {1: "DESS_E1", 2: "DESS_E2"}


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.setLevel(level)
    if not logger.handlers:
        logger.addHandler(handler)


def extract_dicom_archives(
    archive_dir: Path,
    output_dir: Path,
    *,
    skip_existing: bool = True,
) -> tuple[int, int, int]:
    """
    Extract ``{scan_id}.tar.gz`` archives into ``{output_dir}/{scan_id}/``.

    Returns:
        (extracted, skipped, failed) counts.
    """
    archive_dir = Path(archive_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not archive_dir.is_dir():
        logger.warning(f"DICOM archive directory not found: {archive_dir}")
        return 0, 0, 0

    archives = sorted(archive_dir.glob("*.tar.gz"))
    if not archives:
        logger.info(f"No .tar.gz archives found in {archive_dir}")
        return 0, 0, 0

    extracted = skipped = failed = 0
    for archive in archives:
        scan_id = archive.name.removesuffix(".tar.gz")
        dest = output_dir / scan_id
        if skip_existing and dest.is_dir() and any(dest.glob("*.dcm")):
            skipped += 1
            continue

        try:
            with tarfile.open(archive, "r:gz") as tf:
                if hasattr(tarfile, "data_filter"):
                    tf.extractall(output_dir, filter="data")
                else:
                    tf.extractall(output_dir)
            extracted += 1
            logger.debug(f"Extracted {archive.name} -> {dest}")
        except Exception as exc:
            failed += 1
            logger.warning(f"Failed to extract {archive.name}: {exc}")

    logger.info(
        f"DICOM extraction: {extracted} extracted, {skipped} skipped, {failed} failed "
        f"(from {len(archives)} archives in {archive_dir})"
    )
    return extracted, skipped, failed


def _save_slice_last_nifti(
    volume: np.ndarray,
    out_path: Union[str, Path],
    plane: str,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    *,
    from_d_first: bool = True,
) -> tio.ScalarImage:
    img = tio.ScalarImage(
        tensor=np.asarray(volume, dtype=np.float32)[None],
        affine=build_slice_last_affine(plane, spacing),
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return img

def load_annotations(ann_dir: Path) -> pd.DataFrame:
    """Load train/val/test annotation JSONs and build a per-scan label DataFrame.

    Returns:
        DataFrame with columns:
            ID, split, meniscal_tear, ligament_tear, cartilage_lesion, effusion
    """
    rows = []
    for split in ("train", "val", "test"):
        ann_path = ann_dir / f"{split}.json"
        if not ann_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {ann_path}")

        with open(ann_path) as f:
            data = json.load(f)

        cat_map = {c["id"]: c for c in data["categories"]}

        # Build per-image supercategory presence set
        img_supercats: dict[int, set[str]] = {}
        for ann in data["annotations"]:
            cat = cat_map[ann["category_id"]]
            img_supercats.setdefault(ann["image_id"], set()).add(cat["supercategory"])

        for img in data["images"]:
            scan_id = img["scan_id"]
            supercats = img_supercats.get(img["id"], set())
            row = {"ID": scan_id, "split": split}
            for supercat_name, col_name in SUPERCATEGORY_COLUMNS.items():
                row[col_name] = int(supercat_name in supercats)
            rows.append(row)

    df = pd.DataFrame(rows)
    logger.info(f"Loaded annotations: {len(df)} scans across splits")
    for split in ("train", "val", "test"):
        df_s = df[df["split"] == split]
        n_healthy = int((df_s[LABEL_COLUMNS].sum(axis=1) == 0).sum())
        logger.info(f"  {split}: {len(df_s)} scans ({n_healthy} healthy)")
    return df


def dcm_to_nifti(task: dict) -> list[dict] | None:
    """Convert a patient's DICOMs to NIfTI volumes, one per echo.

    Steps:
        1. Read all DICOM headers in the patient directory.
        2. Exclude T2 parameter maps (SeriesDescription contains "T2 map").
        3. Group remaining DICOMs by EchoNumbers (1 → Echo 1, 2 → Echo 2).
        4. For each echo: sort by SliceLocation, stack, save as NIfTI.

    Args:
        task: dict with keys "scan_id", "dcm_dir", "save_dir", "split"

    Returns:
        List of result dicts (one per echo), or None on failure.
    """
    scan_id = task["scan_id"]
    dcm_dir = Path(task["dcm_dir"])
    save_dir = Path(task["save_dir"])
    split = task["split"]

    try:
        dcm_files = sorted(dcm_dir.glob("*.dcm"))
        if not dcm_files:
            logger.warning(f"No DICOM files found for {scan_id} in {dcm_dir}")
            return None

        # Read headers, group by echo number
        echo_groups: dict[int, list[Path]] = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for dcm_path in dcm_files:
                hdr = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
                series_desc = getattr(hdr, "SeriesDescription", "")

                # Skip T2 parameter maps
                if "T2 map" in series_desc or "NOT DIAGNOSTIC" in series_desc:
                    continue

                echo_num = int(getattr(hdr, "EchoNumbers", 0))
                if echo_num in ECHO_TO_SEQ:
                    echo_groups.setdefault(echo_num, []).append(dcm_path)

        if not echo_groups:
            logger.warning(f"No valid DESS DICOMs found for {scan_id}")
            return None

        # Convert DICOMs to NIfTI volumes for each echo
        results = []
        for echo_num in sorted(echo_groups):
            seq_name = ECHO_TO_SEQ[echo_num]
            save_path = save_dir / seq_name /split / "sagittal" / f"{scan_id}.nii.gz"

            slices = []
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for dcm_path in echo_groups[echo_num]:
                    dcm = pydicom.dcmread(str(dcm_path))
                    loc = float(getattr(dcm, "SliceLocation", 0.0))
                    slices.append((loc, dcm.pixel_array.astype(np.float32)))

            slices.sort(key=lambda x: x[0])
            vol = np.stack([s[1] for s in slices], axis=-1)  # (H, W, D)

            # pydicom pixel_array is (Rows, Cols) = (H, W).
            # torchio expects (C, W, H, D), swap H and W before adding channel dim.
            vol = np.swapaxes(vol, 0, 1)  # (H, W, D) → (W, H, D)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            _save_slice_last_nifti(vol, save_path, plane="sagittal")

            results.append({
                "scan_id": scan_id,
                "echo": echo_num,
                "sequence": seq_name,
                "nifti_path": str(save_path),
                "num_slices": vol.shape[-1],
                "size": list(vol.shape),
            })

        return results

    except Exception as e:
        logger.warning(f"Failed to convert {scan_id}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess SKM-TEA dataset: DICOM and annotations to NIfTI and classification labels"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/hpcwork/rwth1833/datasets/SKM-TEA/qdess/v1-release",
        help="Root folder of the SKM-TEA v1 release",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="/hpcwork/rwth1833/datasets/preprocessed/SKM-TEA",
        help="Output directory for NIfTI and metadata CSVs",
    )
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument(
        "--extract-archive",
        action="store_true",
        help="Extract DICOM tarballs before preprocessing.",
    )
    parser.add_argument(
        "--extract-archive-dir",
        type=str,
        default=None,
        help="Directory with {scan_id}.tar.gz DICOM archives "
             "(default: {data_dir}/dicoms when --extract-archive is set).",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)

    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    raw_images_dir = data_dir / "raw_images"
    ann_dir = data_dir / "annotations" / "v1.0.0"

    # ── Step 0: Extract DICOM tarballs (optional one-time setup) ───
    if args.extract_archive:
        archive_dir = (
            Path(args.extract_archive_dir)
            if args.extract_archive_dir is not None
            else data_dir / "dicoms"
        )
        logger.info("=" * 60)
        logger.info("Step 0: Extracting DICOM tarballs")
        logger.info("=" * 60)
        extract_dicom_archives(archive_dir, raw_images_dir)

    # ── Step 1: Load annotations ──────────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 1: Loading annotations")
    logger.info("=" * 60)
    df_labels = load_annotations(ann_dir)

    # ── Step 2: Convert DICOMs to NIfTI ───────────────────────────
    logger.info("=" * 60)
    logger.info("Step 2: Converting qDESS DICOMs to NIfTI (both echoes)")
    logger.info("=" * 60)

    # Build conversion tasks (one per scan)
    tasks = []
    for _, row in df_labels.iterrows():
        scan_id = row["ID"]
        split = row["split"]
        dcm_dir = raw_images_dir / scan_id

        if not dcm_dir.exists():
            logger.warning(f"DICOM directory not found for {scan_id}: {dcm_dir}")
            continue

        tasks.append({
            "scan_id": scan_id,
            "dcm_dir": str(dcm_dir),
            "save_dir": str(save_dir),
            "split": split,
        })

    logger.info(f"Found {len(tasks)} scans to convert (each has 2 echoes)")

    # Parallel conversion of DICOMs to NIfTI
    all_results = []
    with Pool(processes=max(1, args.workers)) as pool:
        chunksize = max(1, len(tasks) // (max(1, args.workers) * 4))
        for result_list in tqdm(
            pool.imap_unordered(dcm_to_nifti, tasks, chunksize=chunksize),
            total=len(tasks),
            desc="Converting DICOMs",
        ):
            if result_list is not None:
                all_results.extend(result_list)

    converted_ids = {r["scan_id"] for r in all_results}
    n_e1 = sum(1 for r in all_results if r["echo"] == 1)
    n_e2 = sum(1 for r in all_results if r["echo"] == 2)
    logger.info(
        f"Successfully converted {len(converted_ids)} scans: "
        f"{n_e1} DESS_E1 and {n_e2} DESS_E2 NIfTI files"
    )

    # Log any failed conversions
    for task in tasks:
        if task["scan_id"] not in converted_ids:
            logger.warning(f"  FAILED: {task['scan_id']}")

    # ── Step 3: Write per-split CSVs ──────────────────────────────
    logger.info("=" * 60)
    logger.info("Step 3: Writing classification CSVs")
    logger.info("=" * 60)

    # Only include scans that were successfully converted
    df_labels = df_labels[df_labels["ID"].isin(converted_ids)]

    for split in ("train", "val", "test"):
        df_split = df_labels[df_labels["split"] == split].copy()
        df_split = df_split.drop(columns=["split"])

        csv_path = save_dir / f"{split}.csv"
        df_split.to_csv(csv_path, index=False)

        n_healthy = int((df_split[LABEL_COLUMNS].sum(axis=1) == 0).sum())
        logger.info(f"  {split}.csv: {len(df_split)} scans ({n_healthy} healthy)")
        for col in LABEL_COLUMNS:
            pos = int(df_split[col].sum())
            neg = len(df_split) - pos
            logger.info(f"    {col}: {pos} positive, {neg} negative")

    logger.info("=" * 60)
    logger.info("Preprocessing completed.")
    logger.info("=" * 60)
    total_nifti = sum(1 for _ in save_dir.rglob("*.nii.gz"))
    logger.info(f"Total NIfTI files: {total_nifti} (2 per scan: DESS_E1 and DESS_E2)")
    logger.info(f"Output directory: {save_dir}")


if __name__ == "__main__":
    main()
