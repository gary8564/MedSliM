"""
Preprocess LUNA25 Challenge dataset for MedSliM.

Converts MetaImage (.mha) full chest CT scans to NIfTI and merges
nodule annotation metadata.

Source layout:
  images/luna25_images/{SeriesInstanceUID}.mha  (4069 CTs)
  annot/LUNA25_Public_Training_Development_Data.csv

Output layout:
  {save_dir}/{SeriesInstanceUID}.nii.gz + metadata.csv
"""
import argparse
import logging
import sys
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


def convert_mha_to_nifti(task: dict) -> dict | None:
    mha_path = Path(task["mha_path"])
    out_path = Path(task["out_path"])

    if out_path.exists():
        logger.debug(f"Skipping (exists): {out_path.name}")
        return None

    try:
        img = sitk.ReadImage(str(mha_path))
        sitk.WriteImage(img, str(out_path))

        series_uid = mha_path.stem
        return {
            "SeriesInstanceUID": series_uid,
            "nifti_path": str(out_path),
            "size": list(img.GetSize()),
            "spacing": list(img.GetSpacing()),
            "origin": list(img.GetOrigin()),
        }
    except Exception as e:
        logger.warning(f"Failed converting {mha_path.name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Preprocess LUNA25: MHA to NIfTI")
    parser.add_argument(
        "--data-dir", type=str,
        default="/hpcwork/rwth1833/datasets/LUNA25",
        help="Root folder containing LUNA25 dataset",
    )
    parser.add_argument(
        "--save-dir", type=str,
        default="/hpcwork/rwth1833/datasets/preprocessed/LUNA25",
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

    # Collect .mha files
    mha_dir = data_dir / "images" / "luna25_images"
    mha_files = sorted(mha_dir.glob("*.mha"))
    logger.info(f"Found {len(mha_files)} MHA files")

    tasks = [
        {"mha_path": str(f), "out_path": str(save_dir / f"{f.stem}.nii.gz")}
        for f in mha_files
    ]

    logger.info("=" * 60)
    logger.info("Converting MHA to NIfTI ...")
    logger.info("=" * 60)

    metadata_rows = []
    with Pool(processes=max(1, args.workers)) as pool:
        for md in tqdm(pool.imap_unordered(convert_mha_to_nifti, tasks), total=len(tasks)):
            if md is not None:
                metadata_rows.append(md)

    # Load annotation CSV and merge with converted file metadata
    annot_csv = data_dir / "annot" / "LUNA25_Public_Training_Development_Data.csv"
    if annot_csv.exists():
        df_annot = pd.read_csv(annot_csv)
        logger.info(f"Loaded {len(df_annot)} annotation entries")

        # Aggregate nodule info per SeriesInstanceUID
        annot_agg = (
            df_annot.groupby("SeriesInstanceUID")
            .agg(
                PatientID=("PatientID", "first"),
                num_nodules=("NoduleID", "nunique"),
                has_malignant=("label", "max"),
                Age=("Age_at_StudyDate", "first"),
                Gender=("Gender", "first"),
            )
            .reset_index()
        )
    else:
        logger.warning(f"Annotation CSV not found: {annot_csv}")
        annot_agg = pd.DataFrame()

    df_meta = pd.DataFrame(metadata_rows)
    if not annot_agg.empty and not df_meta.empty:
        df_meta = df_meta.merge(annot_agg, on="SeriesInstanceUID", how="left")

    df_meta.to_csv(save_dir / "metadata.csv", index=False)

    num_files = len(list(save_dir.glob("*.nii.gz")))
    logger.info(f"Finished. NIfTI files written: {num_files}")
    logger.info("=" * 60)
    logger.info("Preprocessing completed.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
