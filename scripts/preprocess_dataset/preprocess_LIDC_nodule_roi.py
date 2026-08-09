"""
Create LIDC-IDRI nodule ROI crops for localized malignancy evaluation.

This script consumes the outputs from `preprocess_LIDC.py`:

  {input_dir}/annotation.csv
  {input_dir}/nodule_labels.csv
  {input_dir}/malignancy_labels.csv
  {input_dir}/{train,test}/axial/{scan_id}.nii.gz

and writes a new dataset that can be used directly by
`med_slim/utils/preprocessing/precompute_slice_feature.py`:

  {output_dir}/{train,test}/axial/{scan_id}_{nodule_idx}.nii.gz
  {output_dir}/{train,test}_metadata.csv
  {output_dir}/{train,test}_binary.csv
  {output_dir}/{train,test}_multiclass.csv  (malignancy 0..3, remapped from {1,2,4,5})
  {output_dir}/nodule_metadata.csv

The default crop size is 256x256x32, following the MST LIDC crop-or-pad step.
By default, crops are centered with TorchIO's mask-aware `CropOrPad`, using a
union mask reconstructed from the pylidc radiologist annotation masks. 
"""
import argparse
import ast
import logging
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torchio as tio
from tqdm import tqdm

# pylidc 0.2.3 still uses NumPy 1.x aliases removed in NumPy 2.0.
for _alias, _replacement in {
    "int": int,
    "float": float,
    "bool": bool,
    "object": object,
    "complex": complex,
}.items():
    if not hasattr(np, _alias):
        setattr(np, _alias, _replacement)

import pylidc as pl


logger = logging.getLogger(__name__)

# Same remapping as preprocess_LIDC.py: informative ratings {1,2,4,5} -> 0..3.
MALIGNANCY_MULTICLASS_MAP = {1: 0, 2: 1, 4: 2, 5: 3}
MALIGNANCY_CLASS_NAMES = {
    0: "Highly Unlikely",
    1: "Moderately Unlikely",
    2: "Moderately Suspicious",
    3: "Highly Suspicious",
}


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


def parse_bbox(raw_bbox: str | list) -> np.ndarray:
    """
    Parse a pylidc bbox saved by `preprocess_LIDC.py`.

    `ann.bbox()` follows the scan.to_volume() axis order:
      [[axis0_start, axis0_stop], [axis1_start, axis1_stop], [z_start, z_stop]]

    The stop values are Python-style exclusive bounds.
    """
    if isinstance(raw_bbox, list):
        bbox = raw_bbox
    else:
        # CSV rows currently look like:
        # [[np.int64(360), np.int64(383)], ...]
        cleaned = str(raw_bbox).replace("np.int64(", "").replace(")", "")
        bbox = ast.literal_eval(cleaned)

    arr = np.asarray(bbox, dtype=np.int64)
    if arr.shape != (3, 2):
        raise ValueError(f"Expected bbox shape (3, 2), got {arr.shape}: {raw_bbox}")
    return arr


def union_bbox(series: pd.Series) -> np.ndarray:
    boxes = np.stack([parse_bbox(x) for x in series], axis=0)
    return np.stack([boxes[:, :, 0].min(axis=0), boxes[:, :, 1].max(axis=0)], axis=1)


def resolve_source_nifti(input_dir: Path, split: str, scan_id: int) -> Path:
    candidates = [
        input_dir / split / "axial" / f"{scan_id}.nii.gz",
        input_dir / "volumes" / f"{scan_id}.nii.gz",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find source NIfTI for scan_id={scan_id}. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


def crop_or_pad_around_bbox(
    img: nib.Nifti1Image,
    bbox: np.ndarray,
    crop_shape: tuple[int, int, int],
    padding_value: float,
) -> tuple[nib.Nifti1Image, dict]:
    data = np.asanyarray(img.dataobj)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI image, got shape {data.shape}")

    bbox = bbox.astype(np.int64)
    center = ((bbox[:, 0] + bbox[:, 1] - 1) / 2.0).round().astype(np.int64)
    crop_shape_arr = np.asarray(crop_shape, dtype=np.int64)
    start = center - crop_shape_arr // 2
    end = start + crop_shape_arr

    src_start = np.maximum(start, 0)
    src_end = np.minimum(end, np.asarray(data.shape, dtype=np.int64))
    dst_start = src_start - start
    dst_end = dst_start + (src_end - src_start)

    crop = np.full(crop_shape, padding_value, dtype=np.float32)
    crop[
        dst_start[0] : dst_end[0],
        dst_start[1] : dst_end[1],
        dst_start[2] : dst_end[2],
    ] = data[
        src_start[0] : src_end[0],
        src_start[1] : src_end[1],
        src_start[2] : src_end[2],
    ].astype(np.float32, copy=False)

    affine = img.affine.copy()
    affine[:3, 3] = (img.affine @ np.array([start[0], start[1], start[2], 1.0]))[:3]

    out_img = nib.Nifti1Image(crop, affine, header=img.header.copy())
    out_img.header.set_data_dtype(np.float32)

    meta = {
        "crop_start_axis0": int(start[0]),
        "crop_start_axis1": int(start[1]),
        "crop_start_z": int(start[2]),
        "crop_end_axis0": int(end[0]),
        "crop_end_axis1": int(end[1]),
        "crop_end_z": int(end[2]),
        "center_axis0": int(center[0]),
        "center_axis1": int(center[1]),
        "center_z": int(center[2]),
    }
    return out_img, meta


def build_nodule_masks_for_scan(
    scan_id: int, nodule_indices: set[int], shape: tuple[int, int, int]
) -> dict[int, np.ndarray]:
    """
    Reconstruct full-volume union masks from pylidc annotation masks.

    `scan.cluster_annotations()` defines the same nodule_idx ordering used by
    `preprocess_LIDC.py`, so the reconstructed masks align with nodule_labels.csv.
    """
    scan = pl.query(pl.Scan).filter(pl.Scan.id == scan_id).first()
    if scan is None:
        raise ValueError(f"pylidc scan not found for scan_id={scan_id}")

    clusters = scan.cluster_annotations(verbose=False)
    masks: dict[int, np.ndarray] = {}
    for nodule_idx in nodule_indices:
        if nodule_idx >= len(clusters):
            raise IndexError(
                f"scan_id={scan_id} has {len(clusters)} clustered nodules, "
                f"but nodule_idx={nodule_idx} was requested."
            )

        mask = np.zeros(shape, dtype=np.uint8)
        for ann in clusters[nodule_idx]:
            bbox = ann.bbox()
            mask[bbox] |= ann.boolean_mask().astype(np.uint8)
        if not mask.any():
            raise ValueError(f"Empty reconstructed mask for scan_id={scan_id}, nodule_idx={nodule_idx}")
        masks[nodule_idx] = mask
    return masks


def crop_or_pad_around_mask(
    img: nib.Nifti1Image,
    mask: np.ndarray,
    crop_shape: tuple[int, int, int],
    padding_value: float,
) -> tuple[nib.Nifti1Image, nib.Nifti1Image, dict]:
    """
    Mask-centered crop using TorchIO CropOrPad.
    """
    data = np.asanyarray(img.dataobj).astype(np.float32, copy=False)
    if data.ndim != 3:
        raise ValueError(f"Expected a 3D NIfTI image, got shape {data.shape}")
    if mask.shape != data.shape:
        raise ValueError(f"Mask shape {mask.shape} does not match image shape {data.shape}")

    subject = tio.Subject(
        img=tio.ScalarImage(tensor=torch.from_numpy(data[None]), affine=img.affine),
        mask=tio.LabelMap(tensor=torch.from_numpy(mask[None].astype(np.uint8)), affine=img.affine),
    )
    transform = tio.CropOrPad(crop_shape, mask_name="mask", padding_mode=padding_value)
    subject = transform(subject)

    crop_data = subject["img"].tensor[0].numpy().astype(np.float32, copy=False)
    crop_mask = subject["mask"].tensor[0].numpy().astype(np.uint8, copy=False)
    crop_affine = subject["img"].affine

    out_img = nib.Nifti1Image(crop_data, crop_affine, header=img.header.copy())
    out_img.header.set_data_dtype(np.float32)
    out_mask = nib.Nifti1Image(crop_mask, crop_affine)
    out_mask.header.set_data_dtype(np.uint8)

    original_start = np.linalg.inv(img.affine) @ np.array(
        [crop_affine[0, 3], crop_affine[1, 3], crop_affine[2, 3], 1.0]
    )
    start = np.rint(original_start[:3]).astype(np.int64)
    end = start + np.asarray(crop_shape, dtype=np.int64)
    coords = np.argwhere(mask > 0)
    center = np.rint(coords.mean(axis=0)).astype(np.int64)

    meta = {
        "crop_start_axis0": int(start[0]),
        "crop_start_axis1": int(start[1]),
        "crop_start_z": int(start[2]),
        "crop_end_axis0": int(end[0]),
        "crop_end_axis1": int(end[1]),
        "crop_end_z": int(end[2]),
        "center_axis0": int(center[0]),
        "center_axis1": int(center[1]),
        "center_z": int(center[2]),
        "mask_voxels": int(mask.sum()),
    }
    return out_img, out_mask, meta


def process_scan_group(task: tuple) -> list[dict]:
    (
        input_dir,
        output_dir,
        split,
        scan_id,
        records,
        crop_shape,
        padding_value,
        crop_source,
        save_masks,
        overwrite,
    ) = task
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    crop_shape = tuple(crop_shape)
    scan_id = int(scan_id)
    split = str(split)

    src_path = resolve_source_nifti(input_dir, split, scan_id)
    img = nib.load(str(src_path))

    dst_dir = output_dir / split / "axial"
    dst_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = output_dir / split / "mask"
    if save_masks:
        mask_dir.mkdir(parents=True, exist_ok=True)

    masks = None
    if crop_source == "mask":
        masks = build_nodule_masks_for_scan(
            scan_id=scan_id,
            nodule_indices={int(row["nodule_idx"]) for row in records},
            shape=img.shape,
        )

    rows = []
    for row in records:
        nodule_id = str(row["ID"])
        bbox = np.asarray(row["union_bbox"], dtype=np.int64)
        dst_path = dst_dir / f"{nodule_id}.nii.gz"
        if dst_path.exists() and not overwrite:
            raise FileExistsError(f"{dst_path} already exists. Pass --overwrite.")

        mask_path = ""
        if crop_source == "mask":
            assert masks is not None
            roi_img, roi_mask, crop_meta = crop_or_pad_around_mask(
                img,
                mask=masks[int(row["nodule_idx"])],
                crop_shape=crop_shape,
                padding_value=padding_value,
            )
            if save_masks:
                mask_path = str(mask_dir / f"{nodule_id}.nii.gz")
                nib.save(roi_mask, mask_path)
        else:
            roi_img, crop_meta = crop_or_pad_around_bbox(
                img,
                bbox=bbox,
                crop_shape=crop_shape,
                padding_value=padding_value,
            )
        nib.save(roi_img, str(dst_path))

        row_dict = dict(row)
        row_dict.update(crop_meta)
        row_dict["nifti_path"] = str(dst_path)
        row_dict["source_nifti_path"] = str(src_path)
        row_dict["crop_source"] = crop_source
        row_dict["mask_nifti_path"] = mask_path
        rows.append(row_dict)

    return rows


def build_nodule_table(input_dir: Path) -> pd.DataFrame:
    annotation_path = input_dir / "annotation.csv"
    nodule_path = input_dir / "nodule_labels.csv"
    scan_path = input_dir / "malignancy_labels.csv"
    for path in (annotation_path, nodule_path, scan_path):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    df_ann = pd.read_csv(annotation_path)
    df_nodules = pd.read_csv(nodule_path)
    df_scans = pd.read_csv(scan_path, dtype={"ID": str})

    key_cols = ["scan_id", "nodule_idx"]
    df_bbox = (
        df_ann.groupby(key_cols)["bbox"]
        .apply(union_bbox)
        .reset_index(name="union_bbox")
    )

    df = df_nodules.drop(columns=["bbox"], errors="ignore").merge(
        df_bbox, on=key_cols, how="left", validate="one_to_one"
    )
    df = df.merge(
        df_scans[["scan_id", "split"]],
        on="scan_id",
        how="left",
        validate="many_to_one",
    )
    if df["split"].isna().any():
        missing = df[df["split"].isna()][key_cols].head().to_dict("records")
        raise ValueError(f"Some nodules are missing split assignments: {missing}")
    if df["union_bbox"].isna().any():
        missing = df[df["union_bbox"].isna()][key_cols].head().to_dict("records")
        raise ValueError(f"Some nodules are missing annotation bboxes: {missing}")

    df["ID"] = df["scan_id"].astype(str) + "_" + df["nodule_idx"].astype(str)
    bbox_arr = np.stack(df["union_bbox"].to_numpy())
    df["bbox_axis0_start"] = bbox_arr[:, 0, 0]
    df["bbox_axis0_stop"] = bbox_arr[:, 0, 1]
    df["bbox_axis1_start"] = bbox_arr[:, 1, 0]
    df["bbox_axis1_stop"] = bbox_arr[:, 1, 1]
    df["bbox_z_start"] = bbox_arr[:, 2, 0]
    df["bbox_z_stop"] = bbox_arr[:, 2, 1]
    return df


def write_split_csvs(df: pd.DataFrame, output_dir: Path) -> None:
    metadata_cols = [
        "ID",
        "scan_id",
        "nodule_idx",
        "patient_id",
        "split",
        "malignancy",
        "Malignant",
        "nifti_path",
        "source_nifti_path",
        "crop_source",
        "mask_nifti_path",
        "bbox_axis0_start",
        "bbox_axis0_stop",
        "bbox_axis1_start",
        "bbox_axis1_stop",
        "bbox_z_start",
        "bbox_z_stop",
        "center_axis0",
        "center_axis1",
        "center_z",
        "crop_start_axis0",
        "crop_start_axis1",
        "crop_start_z",
        "crop_end_axis0",
        "crop_end_axis1",
        "crop_end_z",
        "mask_voxels",
    ]
    existing_metadata_cols = [col for col in metadata_cols if col in df.columns]
    df[existing_metadata_cols].to_csv(output_dir / "nodule_metadata.csv", index=False)

    for split in ("train", "test"):
        df_split = df[df["split"] == split].copy()
        df_split[["ID", "nifti_path"]].to_csv(output_dir / f"{split}_metadata.csv", index=False)
        df_split[["ID", "Malignant"]].to_csv(
            output_dir / f"{split}_binary.csv", index=False
        )
        df_multiclass = df_split[["ID", "malignancy"]].copy()
        df_multiclass["malignancy"] = df_multiclass["malignancy"].map(MALIGNANCY_MULTICLASS_MAP)
        if df_multiclass["malignancy"].isna().any():
            bad = sorted(df_split.loc[df_multiclass["malignancy"].isna(), "malignancy"].unique())
            raise ValueError(
                f"Unexpected raw malignancy values in {split} split: {bad}. "
                f"Expected informative ratings {sorted(MALIGNANCY_MULTICLASS_MAP)}."
            )
        df_multiclass = df_multiclass.astype({"malignancy": int})
        df_multiclass.to_csv(output_dir / f"{split}_multiclass.csv", index=False)
        counts_b = df_split["Malignant"].value_counts().sort_index().to_dict()
        counts_m = df_multiclass["malignancy"].value_counts().sort_index()
        counts_m_named = {MALIGNANCY_CLASS_NAMES[k]: v for k, v in counts_m.to_dict().items()}
        logger.info(
            "%s: %d nodules, Malignant counts=%s, malignancy=%s",
            split,
            len(df_split),
            counts_b,
            counts_m_named,
        )


def create_roi_dataset(
    input_dir: Path,
    output_dir: Path,
    crop_shape: tuple[int, int, int],
    padding_value: float,
    crop_source: str,
    save_masks: bool,
    overwrite: bool,
    max_nodules: int | None,
    workers: int,
) -> None:
    if output_dir.exists() and overwrite:
        for split in ("train", "test"):
            for subdir in ("axial", "mask"):
                split_dir = output_dir / split / subdir
                if split_dir.exists():
                    shutil.rmtree(split_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = build_nodule_table(input_dir)
    if max_nodules is not None:
        df = df.head(max_nodules).copy()

    rows = []
    grouped = df.groupby(["split", "scan_id"], sort=False)
    tasks = [
        (
            str(input_dir),
            str(output_dir),
            split,
            int(scan_id),
            df_scan.to_dict("records"),
            crop_shape,
            padding_value,
            crop_source,
            save_masks,
            overwrite,
        )
        for (split, scan_id), df_scan in grouped
    ]

    if workers <= 1:
        iterator = map(process_scan_group, tasks)
        for scan_rows in tqdm(iterator, total=len(tasks), desc="Cropping nodule scans"):
            rows.extend(scan_rows)
    else:
        with Pool(processes=int(workers)) as pool:
            for scan_rows in tqdm(
                pool.imap_unordered(process_scan_group, tasks),
                total=len(tasks),
                desc="Cropping nodule scans",
            ):
                rows.extend(scan_rows)

    df_out = pd.DataFrame(rows)
    write_split_csvs(df_out, output_dir)
    logger.info("Wrote ROI dataset to %s", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create nodule-centered LIDC-IDRI ROI crops for localized classification."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/hpcwork/rwth1833/datasets/preprocessed/LIDC-IDRI"),
        help="Preprocessed whole-scan LIDC directory from preprocess_LIDC.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/hpcwork/rwth1833/datasets/preprocessed/LIDC-IDRI-nodules"),
        help="Output directory for nodule ROI NIfTI crops and CSVs.",
    )
    parser.add_argument(
        "--crop-shape",
        type=int,
        nargs=3,
        default=(256, 256, 32),
        metavar=("AXIS0", "AXIS1", "Z"),
        help="Crop-or-pad output shape. Default is 256 256 32.",
    )
    parser.add_argument(
        "--padding-value",
        type=float,
        default=-1024.0,
        help="HU value used when crop extends outside the source volume.",
    )
    parser.add_argument(
        "--crop-source",
        type=str,
        choices=["mask", "bbox"],
        default="mask",
        help="Center crops using reconstructed pylidc masks (MST-style) or union bboxes.",
    )
    parser.add_argument(
        "--save-masks",
        action="store_true",
        help="Also save cropped union masks under {split}/mask/. They are not used by feature precompute.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing ROI crops.")
    parser.add_argument("--max-nodules", type=int, default=None, help="Debug limit.")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker processes.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    create_roi_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        crop_shape=tuple(args.crop_shape),
        padding_value=args.padding_value,
        crop_source=args.crop_source,
        save_masks=args.save_masks,
        overwrite=args.overwrite,
        max_nodules=args.max_nodules,
        workers=max(1, int(args.workers)),
    )


if __name__ == "__main__":
    main()
