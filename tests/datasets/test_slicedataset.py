from pathlib import Path

import pandas as pd
import pytest
import torchio as tio

from med_slim.data.slice_dataset import SliceDataset
from med_slim.utils.preprocessing.transforms import get_transforms
from med_slim.utils.model_config import get_slice_encoder_config

MRNET_ROOT = Path("/hpcwork/rwth1833/datasets/preprocessed/MRNet")
FASTMRI_ROOT = Path("/hpcwork/rwth1833/datasets/preprocessed/fastMRI")
FASTMRI_SEQUENCES = ["pd", "pd_fs", "t2", "t2_fs"]


def _expected_fastMRI_axial_ids(root: Path, split: str, plane: str, sequences: list[str]) -> pd.DataFrame:
    df = pd.read_csv(root / f"{split}.csv", dtype={"ID": str})
    df = df[(df["plane"] == plane) & (df["mri_sequence"].isin(sequences))]
    df = df[df["nifti_path"].map(lambda p: Path(p).is_file())]
    return df.drop_duplicates(subset="ID", keep="last")


@pytest.mark.skipif(not FASTMRI_ROOT.exists(), reason="fastMRI dataset path not available")
def test_slicedataset_fastMRI_multi_sequence():
    split = "train"
    plane = "axial"
    expected = _expected_fastMRI_axial_ids(FASTMRI_ROOT, split, plane, FASTMRI_SEQUENCES)

    ds = SliceDataset(
        path_root=str(FASTMRI_ROOT),
        split=split,
        plane=plane,
        mri_sequences=FASTMRI_SEQUENCES,
    )
    assert len(ds) == len(expected)
    assert set(ds.sample_ids) == set(expected["ID"])

    row = expected.iloc[0]
    assert ds.get_nifti_path(row["ID"]) == Path(row["nifti_path"])
    assert ds.get_nifti_path(row["ID"]).is_file()

    ds_all = SliceDataset(
        path_root=str(FASTMRI_ROOT),
        split=split,
        plane=plane,
        mri_sequences="all",
    )
    assert len(ds_all) == len(ds)
    assert set(ds_all.sample_ids) == set(ds.sample_ids)


@pytest.mark.skipif(not MRNET_ROOT.exists(), reason="MRNet dataset path not available")
def test_slicedataset_shapes_and_metadata():
    split = "train"
    plane = "axial"

    # Load config to derive expected spatial sizes
    model_name = "dinov2"
    cfg = get_slice_encoder_config(model_name)
    H_crop, W_crop = tuple(cfg["img_size"])
    D = 32

    train_tf, _ = get_transforms(model_name=model_name, plane=plane, num_slices=D)

    # Dataset length should match CSV length
    df = pd.read_csv(MRNET_ROOT / f"{split}.csv", index_col="ID")
    ds = SliceDataset(
        path_root=str(MRNET_ROOT),
        split=split,
        transform=train_tf,
        plane=plane,
    )
    assert len(ds) == len(df)

    # Sample one item
    sample = ds[0]
    assert "uid" in sample and "orientation" in sample and "source" in sample
    assert isinstance(sample["source"], tio.ScalarImage)
    assert sample["orientation"] == plane
    assert isinstance(sample["uid"], (str, int))

    # TorchIO Image stores tensor in (C, W, H, D)
    img: tio.ScalarImage = sample["source"]
    tensor = img.data  # same as img.tensor
    assert tensor.ndim == 4
    C, W, H, D_out = tensor.shape
    assert C == 1
    assert W == W_crop and H == H_crop and D_out == D
