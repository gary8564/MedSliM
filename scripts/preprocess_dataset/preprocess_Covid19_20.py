"""
Preprocess COVID-19-20 Lung CT Lesion Segmentation Challenge dataset for MedSliM.

Reorganizes existing NIfTI files into a unified directory structure with
separate ct/ and seg/ subdirectories.

Source layout:
  COVID-19-20_v2/Train/volume-covid19-A-{id}_ct.nii.gz
  COVID-19-20_v2/Train/volume-covid19-A-{id}_seg.nii.gz
  COVID-19-20_v2/Validation/volume-covid19-A-{id}_ct.nii.gz
  test/{id}.nii.gz

Output layout:
  {save_dir}/ct/{split}/{id}.nii.gz
  {save_dir}/seg/{split}/{id}.nii.gz
  metadata.csv
"""
import argparse
import logging
import re
import shutil
import sys
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


def parse_covid_filename(filename: str) -> tuple[str, str] | None:
    """Extract subject ID and type (ct/seg) from COVID-19-20 filename.

    Returns (subject_id, file_type) or None if unparseable.
    """
    # Pattern: volume-covid19-A-{id}_{ct|seg}.nii.gz
    m = re.match(r"volume-covid19-A-(\d+)_(ct|seg)\.nii\.gz", filename)
    if m:
        return m.group(1), m.group(2)
    return None


def collect_tasks(data_dir: Path) -> list[dict]:
    """Collect all NIfTI files from the COVID-19-20 directory structure."""
    tasks = []

    # Train split
    train_dir = data_dir / "COVID-19-20_v2" / "Train"
    if train_dir.exists():
        for f in sorted(train_dir.glob("*.nii.gz")):
            parsed = parse_covid_filename(f.name)
            if parsed:
                subject_id, ftype = parsed
                tasks.append({
                    "source": str(f),
                    "subject_id": subject_id,
                    "file_type": ftype,
                    "split": "train",
                })

    # Validation split
    val_dir = data_dir / "COVID-19-20_v2" / "Validation"
    if val_dir.exists():
        for f in sorted(val_dir.glob("*.nii.gz")):
            parsed = parse_covid_filename(f.name)
            if parsed:
                subject_id, ftype = parsed
                tasks.append({
                    "source": str(f),
                    "subject_id": subject_id,
                    "file_type": ftype,
                    "split": "val",
                })

    # Test split (numeric IDs, CT only, no segmentation)
    test_dir = data_dir / "test"
    if test_dir.exists():
        for f in sorted(test_dir.glob("*.nii.gz")):
            stem = f.name.replace(".nii.gz", "")
            # Handle cases like "189_0" -> subject_id = "189_0"
            tasks.append({
                "source": str(f),
                "subject_id": stem,
                "file_type": "ct",
                "split": "test",
            })

    return tasks


def main():
    parser = argparse.ArgumentParser(description="Preprocess COVID-19-20: reorganize NIfTI files")
    parser.add_argument(
        "--data-dir", type=str,
        default="/hpcwork/rwth1833/datasets/Covid19_20/Post-Challenge",
        help="Root folder containing Post-Challenge data",
    )
    parser.add_argument(
        "--save-dir", type=str,
        default="/hpcwork/rwth1833/datasets/preprocessed/Covid19_20",
        help="Output directory for organized NIfTI files",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Collecting NIfTI files ...")
    tasks = collect_tasks(data_dir)
    logger.info(f"Found {len(tasks)} files")

    metadata_rows = []
    ct_ids_by_split = {}

    for task in tqdm(tasks, desc="Organizing"):
        ftype = task["file_type"]
        split = task["split"]
        subject_id = task["subject_id"]

        out_dir = save_dir / ftype / split
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{subject_id}.nii.gz"

        if not out_path.exists():
            shutil.copy2(task["source"], out_path)

        if ftype == "ct":
            ct_ids_by_split.setdefault(split, set()).add(subject_id)

        metadata_rows.append({
            "ID": subject_id,
            "file_type": ftype,
            "split": split,
            "nifti_path": str(out_path),
        })

    # Build per-subject metadata (one row per CT volume)
    seg_ids = {r["ID"] for r in metadata_rows if r["file_type"] == "seg"}
    ct_rows = [r for r in metadata_rows if r["file_type"] == "ct"]
    for row in ct_rows:
        row["has_segmentation"] = row["ID"] in seg_ids

    df = pd.DataFrame(ct_rows)
    df.to_csv(save_dir / "metadata.csv", index=False)

    logger.info("=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    for split in sorted(ct_ids_by_split):
        n = len(ct_ids_by_split[split])
        logger.info(f"  {split}: {n} CT volumes")
    logger.info(f"Total CT volumes: {len(ct_rows)}")
    logger.info(f"With segmentation: {sum(1 for r in ct_rows if r['has_segmentation'])}")
    logger.info("=" * 60)
    logger.info("Preprocessing completed.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
