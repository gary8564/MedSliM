import os
from pathlib import Path
import pytest
import torchio as tio
import pandas as pd

from med_slim.data.slice_dataset import SliceDataset
from med_slim.utils.preprocessing.transforms import get_transforms, get_model_config

DATA_ROOT = Path("/hpcwork/rwth1833/datasets/preprocessed/MRNet")

@pytest.mark.skipif(not DATA_ROOT.exists(), reason="MRNet dataset path not available")
def test_slicedataset_shapes_and_metadata():
    split = "train"
    plane = "axial"

    # Load config to derive expected spatial sizes
    model_name = "dinov2"
    cfg = get_model_config(model_name)
    H_crop, W_crop = tuple(cfg["img_size"])
    D = 32 

    train_tf, _ = get_transforms(model_name=model_name, num_slices=D)

    # Dataset length should match CSV length
    df = pd.read_csv(DATA_ROOT / f"{split}.csv", index_col="ID")
    ds = SliceDataset(
        path_root=str(DATA_ROOT),
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


