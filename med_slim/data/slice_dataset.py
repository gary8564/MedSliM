import pandas as pd 
import torchio as tio 
import torch.utils.data as data 
import torch
import numpy as np
from pathlib import Path 
from typing import Optional, List

from med_slim.utils.preprocessing import ZNormalization, CropOrPad

def slice_collate_fn(batch):
    """
    Custom collate function to stack TorchIO ScalarImage tensors into [B, C, W, H, D]
    """
    uids = [int(sample["uid"]) for sample in batch]
    orientations = [sample["orientation"] for sample in batch]
    tensors = [sample["source"].tensor for sample in batch]  # each (C, W, H, D)
    x = torch.stack(tensors, dim=0)  # (B, C, W, H, D)
    return {"uid": uids, "orientation": orientations, "x": x}

class SliceDataset(data.Dataset):
    def __init__(
            self,
            path_root: str,
            split: str,
            transform: Optional[tio.Compose] = None,
            plane: str = 'axial',
        ):
        super().__init__()
        if split not in ["train", "val", "test"]:
            raise AttributeError(f"`split` attribute must be a str type and specified as either `train`, `val`, or `test`.")
        if plane not in ["axial", "sagittal", "coronal"]:
            raise AttributeError(f"`plane` attribute must be a str type and specified as either `axial`, `sagittal`, or `coronal`.")
        self.path_root = Path(path_root)
        self.split = split 
        self.transform = transform
        self.df = pd.read_csv(self.path_root/f'{split}.csv', index_col='ID')
        self.sample_ids = self.df.index.tolist()
        self.plane = plane
        
    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        sample_id = self.sample_ids[index]
        uid = int(sample_id)
        img = tio.ScalarImage(self.path_root/f'{self.split}'/f'{self.plane}'/f'{uid:04d}.nii.gz') 
        if self.transform is not None:
            img = self.transform(img)
        return {'uid': uid, "orientation": self.plane, 'source': img}
    
class SliceClassificationDataset(SliceDataset):
    def __init__(
            self,
            path_root: str,
            split: str,
            task: str,
            transform: Optional[tio.Compose] = None,
            labels: Optional[List[str]] = None
        ):
        super().__init__(path_root, split, transform)
        if task not in ["binary", "multiclass", "multilabel"]:
            raise AttributeError(f"`task` attribute must be a str type and specified as either `binary`, `multiclass`, or `multilabel`.")
        if len(labels) > 1 and task == "binary":
            raise AttributeError(f"`labels` attribute must be a list of length 1 for binary classification.")
        self.task = task
        self.labels = labels
        if self.labels is not None:
            self.df = self.df[["ID", *self.labels]]
    
    def __getitem__(self, index):
        sample_id = self.sample_ids[index]
        uid = int(sample_id)
        img = tio.ScalarImage(self.path_root/f'{self.split}'/f'{self.plane}'/f'{uid:04d}.nii.gz') 
        if self.transform is not None:
            img = self.transform(img)
        if self.task == "multilabel":
            target = self.df.loc[sample_id, self.labels].to_numpy(dtype=np.float32) if self.labels is not None else self.df.loc[sample_id].to_numpy(dtype=np.float32)
            target = torch.tensor(target, dtype=torch.float32)
        elif self.task == "binary":
            # For binary classification, BCEWithLogitsLoss expects target shape [B, 1] and float dtype
            target = self.df.loc[sample_id, self.labels[0]] if self.labels is not None else self.df.loc[sample_id, "label"]
            target = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
        else:  # multiclass: if labels provided, one-hot encoding; otherwise, integer encoding
            target = self.df.loc[sample_id, self.labels].to_numpy(dtype=np.int64) if self.labels is not None else self.df.loc[sample_id, "label"].to_numpy(dtype=np.int64)
            target = torch.tensor(target, dtype=torch.long)  
        return {'uid':uid, "orientation": self.plane, 'source': img, 'target':target}

class SliceSegmentationDataset(SliceDataset):
    def __init__(
            self,
            path_root: str,
            split: str,
            transform: Optional[tio.Compose] = None
            ):
        super().__init__(path_root, split, transform)
    
    def __getitem__(self, index):
        sample_id = self.sample_ids[index]
        uid = int(sample_id)
        img = tio.ScalarImage(self.path_root/f'{self.split}'/f'{self.plane}'/f'{uid:04d}.nii.gz')         
        mask = tio.LabelMap(self.path_root/f'{self.split}'/f'{self.plane}'/'mask'/ f'{uid:04d}.nii.gz')
        subject = tio.Subject(img=img, mask=mask)
        if self.transform is not None:
            subject = self.transform(subject)
        img = subject['img']
        mask = subject['mask']
        
        return {'uid':uid, "orientation": self.plane, 'source': img, 'target':mask}