"""
Convert SKM-TEA COCO-style 3D bounding-box annotations to per-pathology ROI
slice ranges (roiZ / roiDepth columns) for slice-attention interpretability.

Reads the original annotation JSONs (train.json, val.json, test.json) and the
existing preprocessed classification CSVs. Outputs augmented CSVs with columns:

    roiZ, roiDepth                  
    roiZ_meniscal_tear,      roiDepth_meniscal_tear
    roiZ_ligament_tear,      roiDepth_ligament_tear
    roiZ_cartilage_lesion,   roiDepth_cartilage_lesion
    roiZ_effusion,           roiDepth_effusion

Scans with no pathology for a given column will have NaN.

Usage:
    python scripts/preprocess_dataset/convert_skmtea_bbox_to_roi.py \
        --ann-dir /path/to/SKM-TEA/qdess/v1-release/annotations/v1.0.0 \
        --csv-dir /path/to/preprocessed/SKM-TEA \
        [--output-dir /path/to/output]   # defaults to csv-dir
"""
import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

SUPERCATEGORY_TO_COL = {
    "Meniscal Tear": "meniscal_tear",
    "Ligament Tear": "ligament_tear",
    "Cartilage Lesion": "cartilage_lesion",
    "Effusion": "effusion",
}
PATHOLOGY_COLS = list(SUPERCATEGORY_TO_COL.values())


def _merge_z_ranges(z_ranges: list[tuple[int, int]]) -> tuple[int, int]:
    """Return (min_z_start, max_z_end) over a list of (z_start, z_end) tuples."""
    z_start = min(r[0] for r in z_ranges)
    z_end = max(r[1] for r in z_ranges)
    return z_start, z_end


def build_roi_dataframe(ann_path: Path) -> pd.DataFrame:
    """
    Parse one COCO annotation JSON and return a DataFrame with per-scan,
    per-pathology ROI slice ranges plus a merged range.

    SKM-TEA bbox format: [x, y, z, dx, dy, dz]
    Z = voxel index along the 3rd volume dimension (LR / RL).

    Negative deltas: some annotators drew boxes in reverse, producing
    negative dx/dy/dz. Normalize so z_start <= z_end.

    RL orientation flip: the preprocessing pipeline sorts DICOM slices
       by ascending SliceLocation.  For LR scans this matches ascending
       InstanceNumber (= annotation Z order), but for RL scans it is the
       reverse.  We flip the Z range for RL scans so that the ROI indices
       align with the preprocessed NIfTI slice ordering.
    """
    with open(ann_path) as f:
        data = json.load(f)

    cat_map = {c["id"]: c for c in data["categories"]}
    id_to_scan = {img["id"]: img["scan_id"] for img in data["images"]}
    id_to_shape = {img["id"]: img["matrix_shape"] for img in data["images"]}
    id_to_orient = {img["id"]: img["orientation"] for img in data["images"]}

    # Collect per-scan, per-pathology Z ranges
    scan_patho_ranges: dict[str, dict[str, list[tuple[int, int]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for ann in data["annotations"]:
        img_id = ann["image_id"]
        scan_id = id_to_scan[img_id]
        supercat = cat_map[ann["category_id"]]["supercategory"]
        col = SUPERCATEGORY_TO_COL.get(supercat)
        if col is None:
            continue

        z = int(ann["bbox"][2])
        dz = int(ann["bbox"][5])
        num_slices = id_to_shape[img_id][2]

        # Normalize: handle negative dz (reverse-drawn boxes)
        z_start = min(z, z + dz)
        z_end = max(z, z + dz) - 1

        # Flip Z for RL scans: annotation Z is in InstanceNumber order,
        # but preprocessed NIfTI is in ascending SliceLocation order which
        # reverses the axis for RL orientation.
        z_orient = id_to_orient[img_id][2]  # "LR" or "RL"
        if z_orient == "RL":
            z_start_flip = num_slices - 1 - z_end
            z_end_flip = num_slices - 1 - z_start
            z_start, z_end = z_start_flip, z_end_flip

        # Clip to valid voxel range [0, num_slices - 1]
        z_start = max(0, z_start)
        z_end = min(num_slices - 1, z_end)

        if z_start > z_end:
            logger.warning(
                f"Skipping degenerate bbox for {scan_id} ({supercat}): "
                f"z={z}, dz={dz} -> [{z_start}, {z_end}]"
            )
            continue

        scan_patho_ranges[scan_id][col].append((z_start, z_end))

    # Healthy scans get NaN ROI columns
    rows = []
    for img in data["images"]:
        scan_id = img["scan_id"]
        row: dict = {"ID": scan_id}

        all_z_ranges: list[tuple[int, int]] = []

        for pcol in PATHOLOGY_COLS:
            ranges = scan_patho_ranges[scan_id].get(pcol)
            if ranges:
                z_start, z_end = _merge_z_ranges(ranges)
                row[f"roiZ_{pcol}"] = z_start
                row[f"roiDepth_{pcol}"] = z_end - z_start + 1
                all_z_ranges.extend(ranges)
            else:
                row[f"roiZ_{pcol}"] = np.nan
                row[f"roiDepth_{pcol}"] = np.nan

        if all_z_ranges:
            z_start, z_end = _merge_z_ranges(all_z_ranges)
            row["roiZ"] = z_start
            row["roiDepth"] = z_end - z_start + 1
        else:
            row["roiZ"] = np.nan
            row["roiDepth"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Convert SKM-TEA bounding boxes to per-pathology roiZ/roiDepth columns"
    )
    parser.add_argument(
        "--ann-dir",
        type=str,
        default="/hpcwork/rwth1833/datasets/SKM-TEA/qdess/v1-release/annotations/v1.0.0",
        help="Directory containing {train,val,test}.json annotation files",
    )
    parser.add_argument(
        "--csv-dir",
        type=str,
        default="/hpcwork/rwth1833/datasets/preprocessed/SKM-TEA",
        help="Directory containing the preprocessed {train,val,test}.csv files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for augmented CSVs (defaults to --csv-dir)",
    )
    args = parser.parse_args()

    ann_dir = Path(args.ann_dir)
    csv_dir = Path(args.csv_dir)
    output_dir = Path(args.output_dir) if args.output_dir else csv_dir

    for split in ("train", "val", "test"):
        ann_path = ann_dir / f"{split}.json"
        csv_path = csv_dir / f"{split}.csv"

        if not ann_path.exists():
            logger.warning(f"Annotation file not found, skipping: {ann_path}")
            continue
        if not csv_path.exists():
            logger.warning(f"CSV file not found, skipping: {csv_path}")
            continue

        # Build ROI DataFrame from bounding boxes
        roi_df = build_roi_dataframe(ann_path)
        logger.info(f"[{split}] Built ROI data for {len(roi_df)} scans from {ann_path.name}")

        # Merge with existing classification CSV
        orig_df = pd.read_csv(csv_path, dtype={"ID": str})

        # Drop any pre-existing ROI columns to avoid conflicts
        roi_cols = [c for c in orig_df.columns if c.startswith("roiZ") or c.startswith("roiDepth")]
        if roi_cols:
            logger.info(f"[{split}] Dropping pre-existing ROI columns: {roi_cols}")
            orig_df = orig_df.drop(columns=roi_cols)

        merged = orig_df.merge(roi_df, on="ID", how="left")

        out_path = output_dir / f"{split}.csv"
        merged.to_csv(out_path, index=False)

        # Report statistics
        n_with_roi = int(merged["roiZ"].notna().sum())
        n_total = len(merged)
        logger.info(f"[{split}] Saved {out_path} — {n_with_roi}/{n_total} scans have ROI annotations")

        for pcol in PATHOLOGY_COLS:
            n = int(merged[f"roiZ_{pcol}"].notna().sum())
            if n > 0:
                depths = merged[f"roiDepth_{pcol}"].dropna()
                logger.info(
                    f"  {pcol:25s}: {n:3d} scans, "
                    f"depth {depths.min():.0f}-{depths.max():.0f} slices "
                    f"(mean {depths.mean():.1f})"
                )

    logger.info("Done.")


if __name__ == "__main__":
    main()
