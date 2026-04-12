"""
Preprocess OASIS-1 (Cross-Sectional) dataset for MedSliM.

Converts Analyze75 (.hdr/.img) processed atlas-registered brain MRI volumes
to NIfTI format.  Uses the T88_111 gain-field-corrected (non-masked) volumes.

Source layout:
  {data_dir}/disc{1-5}/{subject_id}/PROCESSED/MPRAGE/T88_111/
      {subject_id}_mpr_n{3,4}_anon_111_t88_gfc.{hdr,img}

Output layout:
  {save_dir}/{subject_id}.nii.gz + metadata.csv
"""
import argparse
import logging
import sys
import nibabel as nib
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

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


def find_gfc_img(subject_dir: Path) -> Path | None:
    """Find the non-masked gain-field-corrected Analyze75 .img file."""
    t88_dir = subject_dir / "PROCESSED" / "MPRAGE" / "T88_111"
    if not t88_dir.exists():
        return None
    # Match *_111_t88_gfc.img (exclude *_masked_gfc.img)
    candidates = [
        f for f in t88_dir.glob("*_gfc.img")
        if "masked" not in f.name
    ]
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
                "disc": disc_name.name,
            })
    return tasks


def convert_analyze_to_nifti(task: dict, save_dir: Path) -> dict | None:
    subject_id = task["subject_id"]
    out_path = save_dir / f"{subject_id}.nii.gz"

    if out_path.exists():
        logger.debug(f"Skipping (exists): {out_path.name}")
        return None

    try:
        # nibabel transparently loads .hdr/.img pairs via the .img path
        img = nib.load(task["img_path"])
        data = np.asarray(img.dataobj)
        nii = nib.Nifti1Image(data, img.affine, img.header)
        nib.save(nii, str(out_path))

        return {
            "ID": subject_id,
            "nifti_path": str(out_path),
            "disc": task["disc"],
            "shape": list(data.shape),
        }
    except Exception as e:
        logger.warning(f"Failed converting {subject_id}: {e}")
        return None


def load_demographics(data_dir: Path) -> pd.DataFrame:
    """Load cross-sectional demographics xlsx if available."""
    for xlsx in data_dir.glob("oasis_cross-sectional*.xlsx"):
        try:
            df = pd.read_excel(xlsx)
            logger.info(f"Loaded demographics from {xlsx.name} ({len(df)} rows)")
            return df
        except Exception as e:
            logger.warning(f"Could not read {xlsx.name}: {e}")
    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Preprocess OASIS-1: Analyze75 to NIfTI")
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
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Collecting subjects ...")
    tasks = collect_subjects(data_dir)
    logger.info(f"Found {len(tasks)} subjects with processed volumes")

    logger.info("=" * 60)
    logger.info("Converting Analyze75 to NIfTI ...")
    logger.info("=" * 60)

    metadata_rows = []
    for task in tqdm(tasks, desc="Converting"):
        md = convert_analyze_to_nifti(task, save_dir)
        if md is not None:
            metadata_rows.append(md)

    # Merge demographics if available
    df_demo = load_demographics(data_dir)
    df_meta = pd.DataFrame(metadata_rows)

    if not df_demo.empty and not df_meta.empty:
        # The xlsx typically has an "ID" column matching subject_id
        id_col = "ID" if "ID" in df_demo.columns else df_demo.columns[0]
        df_meta = df_meta.merge(df_demo, left_on="ID", right_on=id_col, how="left", suffixes=("", "_demo"))

    df_meta.to_csv(save_dir / "metadata.csv", index=False)

    num_files = len(list(save_dir.glob("*.nii.gz")))
    logger.info(f"Finished. NIfTI files written: {num_files}")
    logger.info("=" * 60)
    logger.info("Preprocessing completed.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
