"""Build study-level labeled train/test CSVs for RSNA-Knee linear probing.

The Kaggle public test set has no abnormality labels. Only 58 training studies
have the 12 structured labels; those labels are constant across series of a
study. This script writes ``train_multilabel.csv`` / ``test_multilabel.csv``
with ``ID=StudyInstanceUID`` so FeatClassificationDataset can resolve one
series per plane via the ``{study}_MR{k}_`` feature-file prefix.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from med_slim.utils.label_metadata import get_dataset_metadata

LABEL_COLUMNS = get_dataset_metadata("RSNA-Knee")["target_labels"]


def build_study_table(series_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(series_csv, dtype={"ID": str, "StudyInstanceUID": str})
    missing = [column for column in LABEL_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{series_csv} is missing label columns: {missing}")
    labeled = df[df[LABEL_COLUMNS].notna().all(axis=1)].copy()
    if labeled.empty:
        raise ValueError(f"No fully labeled rows in {series_csv}")
    studies = (
        labeled.drop_duplicates("StudyInstanceUID", keep="first")
        .loc[:, ["StudyInstanceUID", *LABEL_COLUMNS]]
        .rename(columns={"StudyInstanceUID": "ID"})
        .reset_index(drop=True)
    )
    studies[LABEL_COLUMNS] = studies[LABEL_COLUMNS].astype(int)
    return studies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=Path("/hpcwork/rwth1833/datasets/preprocessed/RSNA-Knee"),
    )
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stratify-label",
        type=str,
        default="ACL",
        help="Label used for a 1-d stratified split (12-way stratify is too sparse).",
    )
    args = parser.parse_args()

    studies = build_study_table(args.annotations_dir / "train.csv")
    if args.stratify_label not in studies.columns:
        raise ValueError(f"Unknown stratify label {args.stratify_label!r}")
    train_df, test_df = train_test_split(
        studies,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=studies[args.stratify_label],
    )
    train_df = train_df.sort_values("ID").reset_index(drop=True)
    test_df = test_df.sort_values("ID").reset_index(drop=True)

    train_path = args.annotations_dir / "train_multilabel.csv"
    test_path = args.annotations_dir / "test_multilabel.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(
        f"Wrote {len(train_df)} train / {len(test_df)} test studies "
        f"(stratify={args.stratify_label}, seed={args.seed}) "
        f"to {train_path} and {test_path}"
    )


if __name__ == "__main__":
    main()
