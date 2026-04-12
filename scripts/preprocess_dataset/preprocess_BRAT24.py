"""
Preprocess BraTS 2024 Challenge dataset for MedSliM.

Copies and reorganizes existing NIfTI files from BraTS-GLI and BraTS-MEN-RT
sub-challenges into a unified per-modality directory structure.

Source:
  BraTS-GLI/{training_data1_v2,...}/{subject_id}/{subject_id}-{modality}.nii.gz
  BraTS-MEN-RT/{Train-v2,...}/{subject_id}/{subject_id}_{modality}.nii.gz

Output:
  {save_dir}/{modality}/{subject_id}.nii.gz + metadata.csv
"""
import argparse
import logging
import shutil
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)

GLI_MODALITIES = {"t1c", "t1n", "t2w", "t2f", "seg"}
MENRT_MODALITIES = {"t1c", "gtv"}


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


def collect_gli_files(gli_dir: Path) -> list[dict]:
    """Collect NIfTI files from all BraTS-GLI data directories."""
    tasks = []
    subdirs = ["training_data1_v2", "training_data_additional", "validation_data"]
    split_map = {
        "training_data1_v2": "train",
        "training_data_additional": "train_additional",
        "validation_data": "val",
    }

    for subdir_name in subdirs:
        subdir = gli_dir / subdir_name
        if not subdir.exists():
            logger.warning(f"BraTS-GLI directory not found: {subdir}")
            continue
        split = split_map[subdir_name]
        for subject_dir in sorted(subdir.iterdir()):
            if not subject_dir.is_dir():
                continue
            subject_id = subject_dir.name
            for modality in GLI_MODALITIES:
                # GLI uses dash separator: {subject_id}-{modality}.nii.gz
                nifti = subject_dir / f"{subject_id}-{modality}.nii.gz"
                if nifti.exists():
                    tasks.append({
                        "source": str(nifti),
                        "subject_id": subject_id,
                        "modality": modality,
                        "subtask": "GLI",
                        "original_split": split,
                    })
    return tasks


def collect_menrt_files(menrt_dir: Path) -> list[dict]:
    """Collect NIfTI files from all BraTS-MEN-RT data directories."""
    tasks = []
    if not menrt_dir.exists():
        logger.warning(f"BraTS-MEN-RT directory not found: {menrt_dir}")
        return tasks

    for candidate in sorted(menrt_dir.iterdir()):
        if not candidate.is_dir():
            continue
        name_lower = candidate.name.lower()

        if "train" in name_lower:
            split = "train"
        elif "val" in name_lower:
            split = "val"
        else:
            # Single-subject directories directly under MEN-RT (e.g. BraTS-MEN-RT-0402-1)
            subject_dir = candidate
            subject_id = subject_dir.name
            for modality in MENRT_MODALITIES:
                nifti = subject_dir / f"{subject_id}_{modality}.nii.gz"
                if nifti.exists():
                    tasks.append({
                        "source": str(nifti),
                        "subject_id": subject_id,
                        "modality": modality,
                        "subtask": "MEN-RT",
                        "original_split": "unknown",
                    })
            continue

        for subject_dir in sorted(candidate.iterdir()):
            if not subject_dir.is_dir():
                continue
            subject_id = subject_dir.name
            for modality in MENRT_MODALITIES:
                # MEN-RT uses underscore separator: {subject_id}_{modality}.nii.gz
                nifti = subject_dir / f"{subject_id}_{modality}.nii.gz"
                if nifti.exists():
                    tasks.append({
                        "source": str(nifti),
                        "subject_id": subject_id,
                        "modality": modality,
                        "subtask": "MEN-RT",
                        "original_split": split,
                    })
    return tasks


def main():
    parser = argparse.ArgumentParser(description="Preprocess BraTS 2024: reorganize NIfTI files")
    parser.add_argument(
        "--data-dir", type=str,
        default="/hpcwork/rwth1833/datasets/BRAT24",
        help="Root folder containing BraTS 2024 sub-challenge directories",
    )
    parser.add_argument(
        "--save-dir", type=str,
        default="/hpcwork/rwth1833/datasets/preprocessed/BRAT24",
        help="Output directory for organized NIfTI files",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Collecting BraTS-GLI files ...")
    logger.info("=" * 60)
    gli_tasks = collect_gli_files(data_dir / "BraTS-GLI")
    logger.info(f"  BraTS-GLI: {len(gli_tasks)} files")

    logger.info("Collecting BraTS-MEN-RT files ...")
    menrt_tasks = collect_menrt_files(data_dir / "BraTS-MEN-RT")
    logger.info(f"  BraTS-MEN-RT: {len(menrt_tasks)} files")

    all_tasks = gli_tasks + menrt_tasks
    logger.info(f"Total files to organize: {len(all_tasks)}")

    logger.info("=" * 60)
    logger.info("Copying files into per-modality directories ...")
    logger.info("=" * 60)

    metadata_rows = []
    for task in tqdm(all_tasks, desc="Copying"):
        modality = task["modality"]
        subject_id = task["subject_id"]
        out_dir = save_dir / modality
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{subject_id}.nii.gz"

        if not out_path.exists():
            shutil.copy2(task["source"], out_path)

        metadata_rows.append({
            "ID": subject_id,
            "modality": modality,
            "subtask": task["subtask"],
            "original_split": task["original_split"],
            "nifti_path": str(out_path),
        })

    df = pd.DataFrame(metadata_rows)
    df.to_csv(save_dir / "metadata.csv", index=False)

    logger.info("=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    for modality in sorted(df["modality"].unique()):
        count = len(df[df["modality"] == modality])
        logger.info(f"  {modality}/: {count} files")
    logger.info(f"Total: {len(df)} files")
    logger.info("=" * 60)
    logger.info("Preprocessing completed.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
