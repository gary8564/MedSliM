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
import warnings
from pathlib import Path
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pydicom
import torchio as tio
from tqdm import tqdm

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
            img = tio.ScalarImage(tensor=vol[None])  # (1, W, H, D)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(str(save_path))

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
        required=True,
        help="Root folder of the SKM-TEA v1 release",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="Output directory for NIfTI and metadata CSVs",
    )
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    args = parser.parse_args()


    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    raw_images_dir = data_dir / "raw_images"
    ann_dir = data_dir / "annotations" / "v1.0.0"

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
