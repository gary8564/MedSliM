import pandas as pd 
import torchio as tio 
import torch.utils.data as data 
import torch
from pathlib import Path 
from typing import Optional

def slice_collate_fn(batch):
    """
    Custom collate function to stack TorchIO ScalarImage tensors into [B, C, W, H, D]
    """
    uids = [sample["uid"] for sample in batch]
    orientations = [sample["orientation"] for sample in batch]
    tensors = [sample["source"].tensor for sample in batch]  # each (C, W, H, D)
    x = torch.stack(tensors, dim=0)  # (B, C, W, H, D)
    return {"uid": uids, "orientation": orientations, "x": x}

class SliceDataset(data.Dataset):
    """
    Dataset for loading 3D medical imaging data as slice sequences.
    
    Directory structure: {path_root}/{split}/{plane}/*.nii.gz
    """
    def __init__(
            self,
            path_root: str,
            split: str,
            transform: Optional[tio.Compose] = None,
            plane: str = 'axial',
        ):
        super().__init__()
        if split not in ["train", "val", "test"]:
            raise AttributeError("`split` attribute must be a str type and specified as either `train`, `val`, or `test`.")
        if plane not in ["axial", "sagittal", "coronal"]:
            raise AttributeError("`plane` attribute must be a str type and specified as either `axial`, `sagittal`, or `coronal`.")
        self.path_root = Path(path_root)
        self.split = split 
        self.transform = transform
        self.plane = plane
        csv_path = self.path_root / f'{split}.csv'
        
        if csv_path.exists():
            self.df = pd.read_csv(csv_path, index_col='ID', dtype={'ID': str})
            if 'plane' in self.df.columns:
                self.df = self.df[self.df['plane'] == plane]
            self.sample_ids = self.df.index.tolist()
        else:
            # Discover from filesystem
            nifti_dir = self.path_root / split / plane
            self.sample_ids = sorted([
                p.stem.replace('.nii', '') 
                for p in nifti_dir.glob('*.nii.gz')
            ])
            self.df = None
        
    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        sample_id = self.sample_ids[index]
        uid = str(sample_id)
        img_path = self.path_root / f'{self.split}' / f'{self.plane}' / f'{uid}.nii.gz'
        img = tio.ScalarImage(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return {'uid': uid, "orientation": self.plane, 'source': img}