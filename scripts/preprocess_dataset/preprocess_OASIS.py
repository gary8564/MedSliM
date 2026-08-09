"""
Preprocess OASIS-1 (Cross-Sectional) dataset for MedSliM.

Converts Analyze75 (.hdr/.img) processed atlas-registered brain MRI volumes
and FSL tissue segmentations to NIfTI. Uses the T88_111 gain-field-corrected
(non-masked) volumes from PROCESSED/ (preferred over RAW: multi-acquisition
aligned, intensity-corrected, Talairach-resampled).

Source layout:
  {data_dir}/disc{1-5}/{subject_id}/PROCESSED/MPRAGE/T88_111/
      {subject_id}_mpr_n{3,4}_anon_111_t88_gfc.{hdr,img}
  {data_dir}/disc{1-5}/{subject_id}/FSL_SEG/
      {subject_id}_*_masked_gfc_fseg.{hdr,img}

Output layout:
  {save_dir}/train/axial/{subject_id}.nii.gz
  {save_dir}/train/seg/{subject_id}_fseg.nii.gz
  {save_dir}/metadata.csv
  {save_dir}/oasis_cross-sectional*.xlsx|pdf  (copied from raw)
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


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


def find_gfc_img(subject_dir: Path) -> Path | None:
    """Find the non-masked gain-field-corrected Analyze75 .img file."""
    t88_dir = subject_dir / "PROCESSED" / "MPRAGE" / "T88_111"
    if not t88_dir.exists():
        return None
    candidates = [
        f for f in t88_dir.glob("*_gfc.img")
        if "masked" not in f.name
    ]
    return candidates[0] if candidates else None


def find_fseg_img(subject_dir: Path) -> Path | None:
    """Find the FSL tissue segmentation Analyze75 .img file."""
    fsl_dir = subject_dir / "FSL_SEG"
    if not fsl_dir.exists():
        return None
    candidates = sorted(fsl_dir.glob("*_fseg.img"))
    return candidates[0] if candidates else None


def collect_subjects(data_dir: Path) -> list[dict]:
    """Collect all subject directories across disc1-disc5."""
    tasks = []
    for disc_name in sorted(data_dir.iterdir()):
        if not disc_name.is_dir() or not disc_name.name.startswith("disc"):
            continue
        for subject_dir in sorted(disc_name.iterdir()):
            if not subject_dir.is_dir() or not subject_dir.name.startswith("OAS1_"):
                continue
            img_file = find_gfc_img(subject_dir)
            if img_file is None:
                logger.warning(f"No processed volume found for {subject_dir.name}")
                continue
            tasks.append({
                "subject_id": subject_dir.name,
                "img_path": str(img_file),
                "fseg_path": (
                    str(p) if (p := find_fseg_img(subject_dir)) is not None else None
                ),
                "disc": disc_name.name,
            })
    return tasks


def convert_analyze_to_nifti(
    img_path: str | Path,
    out_path: Path,
    *,
    force: bool = False,
) -> tuple[bool, list[int] | None]:
    """
    Convert an Analyze75 .img/.hdr pair to NIfTI.

    Returns (wrote_or_exists, shape). shape is None on failure.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not force:
        try:
            img = nib.load(str(out_path))
            shape = list(np.asarray(img.dataobj).shape)
            logger.debug(f"Skipping (exists): {out_path}")
            return False, shape
        except Exception as e:
            logger.warning(f"Existing NIfTI unreadable ({out_path.name}): {e}; reconverting")

    try:
        img = nib.load(str(img_path))
        data = np.asarray(img.dataobj)
        nii = nib.Nifti1Image(data, img.affine, img.header)
        nib.save(nii, str(out_path))
        return True, list(data.shape)
    except Exception as e:
        logger.warning(f"Failed converting {img_path} -> {out_path.name}: {e}")
        return False, None


def copy_demographics_files(data_dir: Path, save_dir: Path) -> list[Path]:
    """Copy OASIS demographics / fact sheets into the preprocessed root."""
    copied: list[Path] = []
    patterns = (
        "oasis_cross-sectional*.xlsx",
        "oasis_cross-sectional*.pdf",
    )
    for pattern in patterns:
        for src in sorted(data_dir.glob(pattern)):
            dst = save_dir / src.name
            if not dst.exists() or dst.stat().st_size != src.stat().st_size:
                shutil.copy2(src, dst)
                logger.info(f"Copied {src.name} -> {dst}")
            else:
                logger.debug(f"Already present: {dst.name}")
            copied.append(dst)
    return copied


def load_demographics(search_dirs: list[Path]) -> pd.DataFrame:
    """
    Load main cross-sectional demographics xlsx (prefer non-reliability table).

    Searches save_dir first (copied files), then data_dir.
    """
    candidates: list[Path] = []
    for d in search_dirs:
        candidates.extend(sorted(d.glob("oasis_cross-sectional*.xlsx")))

    # Prefer the full demographics table over the small reliability sheet.
    preferred = [
        p for p in candidates
        if "reliability" not in p.name.lower()
    ]
    ordered = preferred + [p for p in candidates if p not in preferred]

    for xlsx in ordered:
        try:
            df = pd.read_excel(xlsx)
            logger.info(f"Loaded demographics from {xlsx.name} ({len(df)} rows)")
            return df
        except Exception as e:
            logger.warning(f"Could not read {xlsx.name}: {e}")
    return pd.DataFrame()


def build_metadata_row(
    task: dict,
    volume_path: Path,
    seg_path: Path | None,
    shape: list[int] | None,
) -> dict:
    row = {
        "ID": task["subject_id"],
        "nifti_path": str(volume_path),
        "seg_path": str(seg_path) if seg_path is not None and seg_path.exists() else "",
        "disc": task["disc"],
        "shape": shape if shape is not None else "",
    }
    return row


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess OASIS-1: Analyze75 MRI + FSL_SEG to NIfTI"
    )
    parser.add_argument(
        "--data-dir", type=str,
        default="/hpcwork/rwth1833/datasets/OASIS",
        help="Root folder containing OASIS disc directories",
    )
    parser.add_argument(
        "--save-dir", type=str,
        default="/hpcwork/rwth1833/datasets/preprocessed/OASIS",
        help="Output directory for NIfTI files and metadata",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Reconvert even if output NIfTI already exists",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    volume_dir = save_dir / "train" / "axial"
    seg_dir = save_dir / "train" / "seg"
    volume_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Copying demographics / fact sheets ...")
    copy_demographics_files(data_dir, save_dir)

    logger.info("Collecting subjects ...")
    tasks = collect_subjects(data_dir)
    logger.info(f"Found {len(tasks)} subjects with processed volumes")

    logger.info("=" * 60)
    logger.info("Converting MRI (PROCESSED T88_111 gfc) -> train/axial/")
    logger.info("=" * 60)

    metadata_rows = []
    n_vol_new = n_seg_new = 0
    for task in tqdm(tasks, desc="Converting"):
        subject_id = task["subject_id"]
        vol_out = volume_dir / f"{subject_id}.nii.gz"
        wrote, shape = convert_analyze_to_nifti(
            task["img_path"], vol_out, force=args.force
        )
        if wrote:
            n_vol_new += 1
        if shape is None and not vol_out.exists():
            logger.warning(f"Skipping metadata for failed volume {subject_id}")
            continue

        seg_out: Path | None = None
        if task["fseg_path"] is not None:
            seg_out = seg_dir / f"{subject_id}_fseg.nii.gz"
            wrote_seg, _ = convert_analyze_to_nifti(
                task["fseg_path"], seg_out, force=args.force
            )
            if wrote_seg:
                n_seg_new += 1
        else:
            logger.warning(f"No FSL_SEG for {subject_id}")

        metadata_rows.append(
            build_metadata_row(task, vol_out, seg_out, shape)
        )

    df_meta = pd.DataFrame(metadata_rows)
    df_demo = load_demographics([save_dir, data_dir])

    if not df_demo.empty and not df_meta.empty:
        id_col = "ID" if "ID" in df_demo.columns else df_demo.columns[0]
        # Avoid duplicate ID column from a right-side rename collision.
        demo = df_demo.copy()
        if id_col != "ID":
            demo = demo.rename(columns={id_col: "ID"})
        overlap = [c for c in demo.columns if c in df_meta.columns and c != "ID"]
        if overlap:
            demo = demo.drop(columns=overlap)
        df_meta = df_meta.merge(demo, on="ID", how="left")
        n_matched = int(df_meta["ID"].isin(demo["ID"]).sum())
        logger.info(f"Merged demographics: {n_matched}/{len(df_meta)} IDs matched")

    meta_path = save_dir / "metadata.csv"
    df_meta.to_csv(meta_path, index=False)

    n_vol = len(list(volume_dir.glob("*.nii.gz")))
    n_seg = len(list(seg_dir.glob("*.nii.gz")))
    logger.info(f"Wrote {n_vol_new} new volumes; total volumes: {n_vol}")
    logger.info(f"Wrote {n_seg_new} new segs; total segs: {n_seg}")
    logger.info(f"metadata.csv -> {meta_path} ({len(df_meta)} rows)")
    logger.info("=" * 60)
    logger.info("Preprocessing completed.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
