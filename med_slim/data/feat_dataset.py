import numpy as np
import torch
import random
import os
import logging
from typing import List, Tuple, Dict
from torch.utils.data import Dataset
from glob import glob
from tqdm import tqdm
from safetensors import safe_open
from collections import defaultdict

from med_slim.logging.setup import init_logging
init_logging()
logger = logging.getLogger(__name__)

class PrecomputedFeatPairDataset(Dataset):
    """
    Dataset of precomputed feature pairs from different slice encoder models and view planes, which are used for contrastive learning.
    For multi-plane views, there are two options to handle here:
    (1) Same-plane positives: when forming a pair, enforce both views to be from the same plane; use all planes across the dataset,
    but don't mixed within a pair.
    #TODO
    (2) Mixed-plane positives: use SupCon loss (multi-positive loss) so that the loss function will treat every every view of the same exam (axial, sagittal, coronal) as separate positives in the denominator,
    explicitly tells the model: 'all these are positives for this exam'.
    """
    def __init__(self, 
                 feat_dirs: List[Dict[str, str]], 
                 slice_encoder_models: List[str], 
                 view_planes: List[str], 
                 split: str, 
                 max_feature_dim: int):
        self.feat_dirs = feat_dirs
        self.slice_encoder_models = slice_encoder_models
        self.view_planes = view_planes
        self.split = split
        self.max_feature_dim = max_feature_dim
        self.feat_path_dict = self._get_feat_path_dict_by_study_id()
        self.study_ids = list(self.feat_path_dict.keys())
        
    def _get_feat_path_dict_by_study_id(self):
        feat_path_dict = defaultdict(lambda: defaultdict(list))
        logger.info(f'Selected slice encoder models: {self.slice_encoder_models}')
        logger.info(f'Selected view planes: {self.view_planes}')
        for dataset in tqdm(self.feat_dirs, leave=False):
            dataset_name, feat_dir = dataset["name"], dataset["feat_dir"]
            for model_name in tqdm(self.slice_encoder_models, leave=False):
                for view_plane in tqdm(self.view_planes, leave=False):
                    feat_path = os.path.join(feat_dir, model_name, self.split, view_plane)
                    feat_files = glob(os.path.join(feat_path, "*.safetensors"))
                    assert len(feat_files) > 0, f"couldnt find any feat files in path {feat_path} for model {model_name} and view plane {view_plane}!"
                    for feat_file in feat_files:
                        exam_id = int(os.path.basename(feat_file).split(".")[0])
                        feat_path_dict[f"{dataset_name}_{exam_id}"][view_plane].append(feat_file)
        return feat_path_dict

    def _pad_embed_dim(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pad the embedding dimension to the largest embedding dimension in the batch by padding with zeros if it is smaller than the target dimension
        so that the DataLoader collate can concatenate the tensors of the same shape per batch.
        """
        num_slices, embed_dim = x.shape
        if embed_dim < self.max_feature_dim:
            pad_size = self.max_feature_dim - embed_dim
            x = torch.cat([x, torch.zeros(num_slices, pad_size, device=x.device)], dim=1)
        return x, embed_dim
    
    def _load_feats(self, feat_path: str) -> Tuple[torch.Tensor, dict]:
        with safe_open(feat_path, framework="pt", device="cpu") as f:
            feats = f.get_tensor("feats")
            metadata = f.metadata()
        assert feats.ndim == 2, f"Expected number of dimensions to be 2, but got {feats.ndim=}!"
        assert metadata["plane"] in self.view_planes, f"Expected plane to be in {self.view_planes}, but got {metadata['plane']}!"
        assert metadata["model_name"] in self.slice_encoder_models, f"Expected model name to be in {self.slice_encoder_models}, but got {metadata['model_name']}!"
        return feats, metadata
    
    def __len__(self):
        return len(self.study_ids)

    def __getitem__(self, idx):
        study_id = self.study_ids[idx]
        selected_view_plane = random.choice(self.view_planes)
        assert len(self.feat_path_dict[study_id][selected_view_plane]) == len(self.slice_encoder_models)
        idx1 = np.random.randint(0, len(self.slice_encoder_models))
        idx2 = np.random.randint(0, len(self.slice_encoder_models))
        feat_path1 = self.feat_path_dict[study_id][selected_view_plane][idx1]
        feat_path2 = self.feat_path_dict[study_id][selected_view_plane][idx2]
        feats1, metadata1 = self._load_feats(feat_path1)
        feats2, metadata2 = self._load_feats(feat_path2)
        assert metadata1["num_slices"] == metadata2["num_slices"], \
            f"Expected number of slices to be equal, but got {metadata1['num_slices']} and {metadata2['num_slices']}!"
        assert metadata1["plane"] == metadata2["plane"], \
            f"Expected plane to be equal, but got {metadata1['plane']} and {metadata2['plane']}!"
        with torch.no_grad():
            feats1, orig_embed_dim1 = self._pad_embed_dim(feats1.clone().detach())
            feats2, orig_embed_dim2 = self._pad_embed_dim(feats2.clone().detach())    
        assert feats1.shape == feats2.shape, f"Expected shapes to be equal, but got {feats1.shape} and {feats2.shape}!"
        
        return feats1, torch.as_tensor(orig_embed_dim1), feats2, torch.as_tensor(orig_embed_dim2)
