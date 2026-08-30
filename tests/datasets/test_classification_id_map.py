import pandas as pd

from med_slim.data.feat_dataset import index_plane_feat_files, resolve_classification_id_map
from med_slim.utils.label_metadata import get_annotation_paths_by_split, get_dataset_metadata


def test_resolve_exact_series_ids_and_drop_nan():
    labels = pd.DataFrame(
        {
            "ACL": [1.0, 0.0, float("nan")],
            "Effusion": [1.0, 0.0, 1.0],
        },
        index=["series_a", "series_b", "series_c"],
    )
    feat_map = {
        "series_a": "series_a.safetensors",
        "series_b": "series_b.safetensors",
        "series_c": "series_c.safetensors",
        "unlabeled": "unlabeled.safetensors",
    }
    resolved = resolve_classification_id_map(labels, feat_map, ["ACL", "Effusion"])
    assert resolved == {
        "series_a": "series_a.safetensors",
        "series_b": "series_b.safetensors",
    }


def test_resolve_study_uid_picks_lowest_mr_index():
    study = "1.2.826.0.1.3680043.8.498.studyA"
    labels = pd.DataFrame({"ACL": [1], "Effusion": [0]}, index=[study])
    feat_map = {
        f"{study}_MR2_seriesZ": f"{study}_MR2_seriesZ.safetensors",
        f"{study}_MR0_seriesX": f"{study}_MR0_seriesX.safetensors",
        f"{study}_MR10_seriesY": f"{study}_MR10_seriesY.safetensors",
    }
    resolved = resolve_classification_id_map(labels, feat_map, ["ACL", "Effusion"])
    assert resolved[study] == f"{study}_MR0_seriesX.safetensors"


def test_rsna_knee_eval_metadata():
    meta = get_dataset_metadata("RSNA-Knee")
    assert meta["task"] == "multilabel"
    assert len(meta["target_labels"]) == 12
    assert "Baker's" in meta["target_labels"]


def test_multilabel_annotation_path_prefers_suffix(tmp_path):
    (tmp_path / "train.csv").write_text("ID\nall\n")
    (tmp_path / "train_multilabel.csv").write_text("ID\nlabeled\n")
    (tmp_path / "test_multilabel.csv").write_text("ID\nholdout\n")
    paths = get_annotation_paths_by_split(
        str(tmp_path), "multilabel", splits=["train", "test"]
    )
    assert paths["train"].endswith("train_multilabel.csv")
    assert paths["test"].endswith("test_multilabel.csv")


def test_index_plane_feat_files_prefers_split_then_sibling(tmp_path):
    train_dir = tmp_path / "mri-core" / "train" / "sagittal"
    test_dir = tmp_path / "mri-core" / "test" / "sagittal"
    train_dir.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    holdout = "1.2.826.0.1.3680043.8.498.holdout_MR0_1.2.826.0.1.3680043.8.498.series"
    (train_dir / f"{holdout}.safetensors").write_bytes(b"")
    (train_dir / "trainonly.safetensors").write_bytes(b"")
    (test_dir / "official_test.safetensors").write_bytes(b"")

    index = index_plane_feat_files(
        str(tmp_path), "mri-core", "sagittal", preferred_split="test"
    )
    assert index["official_test"] == ("test", "official_test.safetensors")
    assert index[holdout] == ("train", f"{holdout}.safetensors")
    assert index["trainonly"] == ("train", "trainonly.safetensors")
