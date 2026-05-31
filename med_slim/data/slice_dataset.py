import pandas as pd 
import torchio as tio 
import torch.utils.data as data 
import torch
from pathlib import Path 
from typing import Dict, List, Optional, Sequence, Union

from med_slim.utils.preprocessing import CropEmptySlices, EnsureSliceAxisLast

VIEW_PLANES = frozenset({"axial", "sagittal", "coronal"})


def _nifti_uid(path: Path) -> str:
    """Return the sample UID stored in precomputed feature caches (stem without .nii.gz)."""
    return path.name.removesuffix(".nii.gz")


def _find_mri_sequences_folder(path_root: Path, split: str, plane: str) -> List[str]:
    """Find MRI sequence folders under ``{path_root}/{split}/``."""
    split_dir = path_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    sequences = []
    for child in sorted(split_dir.iterdir()):
        if not child.is_dir() or child.name in VIEW_PLANES:
            continue
        seq_plane_dir = child / plane
        if seq_plane_dir.is_dir() and any(seq_plane_dir.glob("*.nii.gz")):
            sequences.append(child.name)

    if not sequences:
        raise FileNotFoundError(
            f"No MRI sequence folders with {plane} NIfTI files under {split_dir}. "
            f"Expected {{split}}/{{sequence}}/{{plane}}/*.nii.gz."
        )
    return sequences


def _resolve_mri_sequences(
    path_root: Path,
    split: str,
    plane: str,
    mri_sequences: Optional[Union[str, Sequence[str]]],
) -> Optional[List[str]]:
    if mri_sequences is None:
        return None

    sequences = (
        [mri_sequences] if isinstance(mri_sequences, str) else list(mri_sequences)
    )
    if not sequences:
        raise ValueError("mri_sequences must contain at least one sequence name when provided.")

    if len(sequences) == 1 and sequences[0].lower() == "all":
        return _find_mri_sequences_folder(path_root, split, plane)

    missing = [seq for seq in sequences if not (path_root / split / seq / plane).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Missing MRI sequence directories for plane '{plane}': "
            + ", ".join(str(path_root / split / seq / plane) for seq in missing)
        )
    return sequences


def _index_multi_sequence_paths(
    path_root: Path,
    split: str,
    plane: str,
    sequences: Sequence[str],
) -> Dict[str, Path]:
    """Union NIfTI paths from {split}/{sequence}/{plane}/ into dictionary {UID: Path}."""
    uid_to_path: Dict[str, Path] = {}
    duplicates: Dict[str, List[Path]] = {}

    for sequence in sequences:
        seq_dir = path_root / split / sequence / plane
        for nifti_path in sorted(seq_dir.glob("*.nii.gz")):
            uid = _nifti_uid(nifti_path)
            if uid in uid_to_path:
                duplicates.setdefault(uid, [uid_to_path[uid]]).append(nifti_path)
                continue
            uid_to_path[uid] = nifti_path

    if duplicates:
        uid, paths = next(iter(duplicates.items()))
        raise ValueError(
            f"Duplicate NIfTI UID '{uid}' found in multiple MRI sequence folders: "
            + ", ".join(str(p) for p in paths)
        )

    if not uid_to_path:
        raise FileNotFoundError(
            f"No NIfTI files found under "
            f"{{split}}/{{sequence}}/{{plane}} for sequences={list(sequences)}."
        )
    return uid_to_path


def slice_collate_fn(batch):
    """
    Custom collate function to stack TorchIO ScalarImage tensors into [B, C, W, H, D]
    """
    uids = [sample["uid"] for sample in batch]
    orientations = [sample["orientation"] for sample in batch]
    tensors = [sample["source"].tensor for sample in batch]  # each (C, W, H, D)
    x = torch.stack(tensors, dim=0)  # (B, C, W, H, D)
    return {"uid": uids, "orientation": orientations, "x": x}


def generate_tiled_images(image: tio.ScalarImage, grid_size: int):
    """
    Build a global view plus tile views before FM preprocessing.

    The image is already canonicalized and has slice axis last, but has not yet gone
    through FM-specific resize/crop and normalization. This keeps tiling simple:
    split the original-resolution in-plane tensor, then run the normal preprocessing
    transform on every view.

    Returns:
        (images, region_boxes), ordered as [global, row-major tiles].
    """
    data = image.tensor
    _, W, H, _ = data.shape
    affine = image.affine

    images = [tio.ScalarImage(tensor=data.clone(), affine=affine)]
    region_boxes = [[0.0, 0.0, 1.0, 1.0]]

    # Integer tile boundaries (robust to resolutions not divisible by grid_size).
    w_edges = torch.linspace(0, W, grid_size + 1).round().to(torch.long).tolist()
    h_edges = torch.linspace(0, H, grid_size + 1).round().to(torch.long).tolist()

    # Regional tiles (row-major over the original-resolution grid).
    for r in range(grid_size):
        for c in range(grid_size):
            w0, w1 = w_edges[r], w_edges[r + 1]
            h0, h1 = h_edges[c], h_edges[c + 1]
            tile = data[:, w0:w1, h0:h1, :].contiguous()
            images.append(tio.ScalarImage(tensor=tile, affine=affine))
            region_boxes.append([w0 / W, h0 / H, w1 / W, h1 / H])

    return images, region_boxes


def tiled_slice_collate_fn(batch):
    """Stack tiled samples into [B, num_tiled_regions, C, W, H, num_slices]."""
    uids = [sample["uid"] for sample in batch]
    orientations = [sample["orientation"] for sample in batch]
    tensors = [sample["source"] for sample in batch]  # each [num_tiled_regions, C, W, H, num_slices]
    x = torch.stack(tensors, dim=0)
    return {
        "uid": uids,
        "orientation": orientations,
        "x": x,
        "region_boxes": [sample["region_boxes"] for sample in batch],
    }


def build_pre_tile_transform(plane: str, crop_empty_slices: bool = False):
    """Prepare a volume for original-resolution tiling without FM resize/normalization."""
    transforms = [
        tio.ToCanonical(),
        EnsureSliceAxisLast(plane=plane),
    ]
    if crop_empty_slices:
        transforms.append(CropEmptySlices())
    return tio.Compose(transforms)


class SliceDataset(data.Dataset):
    """
    Dataset for loading 3D medical imaging data as slice sequences.

    Default directory structure:
        {path_root}/{split}/{plane}/*.nii.gz

    Multi-sequence directory structure:
        {path_root}/{split}/{sequence}/{plane}/*.nii.gz
        
    If all series for different MRI sequences are unique, the precomputed features are unioned to the same output directory sorted by train/test split and view plane.
    If not, the error will be raised.
    """
    def __init__(
            self,
            path_root: str,
            split: str,
            transform: Optional[tio.Compose] = None,
            plane: str = 'axial',
            mri_sequences: Optional[Union[str, Sequence[str]]] = None,
        ):
        super().__init__()
        if split not in ["train", "val", "test"]:
            raise AttributeError("`split` attribute must be a str type and specified as either `train`, `val`, or `test`.")
        if plane not in VIEW_PLANES:
            raise AttributeError("`plane` attribute must be a str type and specified as either `axial`, `sagittal`, or `coronal`.")
        self.path_root = Path(path_root)
        self.split = split 
        self.transform = transform
        self.plane = plane
        self.mri_sequences = _resolve_mri_sequences(
            self.path_root, split, plane, mri_sequences
        )
        self._uid_to_path: Optional[Dict[str, Path]] = None
        csv_path = self.path_root / f'{split}.csv'

        if self.mri_sequences is not None:
            self.df = None
            if csv_path.exists():
                df = pd.read_csv(csv_path, dtype={"ID": str})
                if "plane" in df.columns:
                    df = df[df["plane"] == plane]
                if "mri_sequence" in df.columns:
                    df = df[df["mri_sequence"].isin(self.mri_sequences)]
                if "nifti_path" not in df.columns:
                    raise ValueError(
                        f"{csv_path} must contain a 'nifti_path' column when mri_sequences is set."
                    )
                self._uid_to_path = {
                    str(row["ID"]): Path(row["nifti_path"])
                    for _, row in df.iterrows()
                    if Path(row["nifti_path"]).is_file()
                }
                self.sample_ids = sorted(self._uid_to_path)
            else:
                self._uid_to_path = _index_multi_sequence_paths(
                    self.path_root, split, plane, self.mri_sequences
                )
                self.sample_ids = sorted(self._uid_to_path)
        elif csv_path.exists():
            df = pd.read_csv(csv_path, dtype={"ID": str})
            if "plane" in df.columns:
                df = df[df["plane"] == plane]
            if "nifti_path" in df.columns:
                self._uid_to_path = {
                    str(row["ID"]): Path(row["nifti_path"])
                    for _, row in df.iterrows()
                    if Path(row["nifti_path"]).is_file()
                }
                self.sample_ids = sorted(self._uid_to_path)
                self.df = None
            else:
                self.df = df.set_index("ID")
                self.sample_ids = self.df.index.tolist()
        else:
            nifti_dir = self.path_root / split / plane
            self.sample_ids = sorted(_nifti_uid(p) for p in nifti_dir.glob('*.nii.gz'))
            self.df = None
        
    def __len__(self):
        return len(self.sample_ids)

    def get_nifti_path(self, uid: str) -> Path:
        """Return the NIfTI path for a sample UID."""
        uid = str(uid)
        if self._uid_to_path is not None:
            if uid not in self._uid_to_path:
                raise KeyError(f"UID '{uid}' not found in indexed MRI sequence paths.")
            return self._uid_to_path[uid]
        return self.path_root / self.split / self.plane / f"{uid}.nii.gz"

    def __getitem__(self, index):
        sample_id = self.sample_ids[index]
        uid = str(sample_id)
        img = tio.ScalarImage(self.get_nifti_path(uid))
        if self.transform is not None:
            img = self.transform(img)
        return {'uid': uid, "orientation": self.plane, 'source': img}


class TiledSliceDataset(SliceDataset):
    """Slice dataset that creates global and regional views."""

    def __init__(self, *args, pre_tile_transform, view_transform, grid_size: int, **kwargs):
        super().__init__(*args, transform=pre_tile_transform, **kwargs)
        self.view_transform = view_transform
        self.grid_size = grid_size

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        images, region_boxes = generate_tiled_images(sample["source"], self.grid_size)
        views = []
        for image in images:
            transformed = self.view_transform(image)
            views.append(transformed.tensor)
        return {
            "uid": sample["uid"],
            "orientation": sample["orientation"],
            "source": torch.stack(views, dim=0),
            "region_boxes": region_boxes,
        }
