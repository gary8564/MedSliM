import numpy as np
import pandas as pd
import torch
import random
import os
import logging
from typing import List, Tuple, Dict, Optional
from torch.utils.data import Dataset
from glob import glob
from tqdm import tqdm
from safetensors import safe_open
from collections import defaultdict

from med_slim.logging.setup import init_logging
init_logging()
logger = logging.getLogger(__name__)


def linear_classifier_collate_fn(batch):
    """
    Custom collate function to handle list of variable-length feature embeddings from different slice encoders.
    """
    targets_list = [item["label"] for item in batch]
    targets = torch.tensor(targets_list, dtype=batch[0]["label"].dtype)
    sample_ids_list = [item["sample_id"] for item in batch]
    
    all_feats_list = [item["feature_embeds"] for item in batch]
    K = len(all_feats_list[0]) # K = number of foundation models
    # Create a list to store the K collated feature batches
    collated_feats_list = []
    for k in range(K):
        # Gather the k-th feature tensor from all N samples in the batch
        kth_feature_tensors = [sample_feats[k] for sample_feats in all_feats_list]

        # Stack these N tensors along a new batch dimension (dimension 0),
        # resulting in a batch tensor of shape [N, num_slices, embed_dim]
        kth_feature_batch = torch.stack(kth_feature_tensors, dim=0)
        collated_feats_list.append(kth_feature_batch)
    
    return {
        "feature_embeds": collated_feats_list,
        "labels": targets,
        "sample_ids": sample_ids_list,
    }


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


class FeatClassificationDataset(Dataset):
    """
    Dataset for downstream classification tasks using precomputed slice features.
    This dataset loads precomputed features and their corresponding labels for 
    linear probing evaluation of the SSL model.
    
    When multiple slice_encoder_models are provided, returns features from all encoders.
    """
    def __init__(self,
                 feat_dir: str,
                 slice_encoder_models: List[str],
                 view_plane: str,
                 split: str,
                 annotations_path: str,
                 task: str,
                 target_columns: List[str]):
        """
        Args:
            feat_dir: Directory containing precomputed features organized as:
                      feat_dir/{model_name}/{split}/{plane}/*.safetensors
            slice_encoder_models: List of slice encoder model names (e.g., ["dinov2", "rad-dino"])
                                  If single model, pass as ["dinov2"]
            view_plane: Which view plane to use (e.g., "sagittal", "axial", "coronal")
            split: Data split ("train" or "test")
            annotations_path: Path to the annotation CSV file with columns: ID, target1, target2, ...
            task: Classification task type ("binary", "multiclass", or "multilabel")
            target_columns: List of column names to use as classification targets
                           For binary/multiclass: single column.
                           For multilabel: multiple columns.
        """
        # Validate task
        if task not in ["binary", "multiclass", "multilabel"]:
            raise ValueError(f"`task` must be 'binary', 'multiclass', or 'multilabel', got '{task}'")
        if task in ["binary", "multiclass"] and len(target_columns) != 1: # multiclass in medical domain is usually ordinal data --> integer encoding
            raise ValueError(f"For {task} classification, `target_columns` must have exactly 1 element")
        
        self.task = task
        self.target_columns = target_columns
        self.feat_dir = feat_dir
        self.slice_encoder_models = slice_encoder_models
        self.view_plane = view_plane
        self.split = split
        
        # Load labels
        self.df_labels = pd.read_csv(annotations_path)
        self.df_labels.set_index("ID", inplace=True)
        self.sample_ids = list(self.df_labels.index.astype(int))
        
        # Build feature paths for all encoders and verify consistency
        self.feat_paths = {}
        self.id_filename_map = {}
        
        for model_name in self.slice_encoder_models:
            feat_path = os.path.join(feat_dir, model_name, split, view_plane)
            if not os.path.exists(feat_path):
                raise FileNotFoundError(f"Feature path {feat_path} does not exist!")
            self.feat_paths[model_name] = feat_path
            
            feat_files = glob(os.path.join(feat_path, '*.safetensors'))
            if len(feat_files) != len(self.sample_ids):
                raise ValueError(f"Expected {len(self.sample_ids)} feature files for {model_name}, but got {len(feat_files)}!")
            
            # Build id->filename mapping from first encoder (filenames should match across encoders)
            if not self.id_filename_map:
                for f in feat_files:
                    fname = os.path.basename(f)
                    fid = int(fname.split(".")[0])
                    self.id_filename_map[fid] = fname
        
    def __len__(self):
        return len(self.sample_ids)
    
    def _load_feat(self, feat_path: str) -> Tuple[torch.Tensor, dict]:
        with safe_open(feat_path, framework="pt", device="cpu") as f:
            feat = f.get_tensor("feats")
            metadata = f.metadata()
        return feat, metadata
    
    def _get_label(self, sample_id: int) -> torch.Tensor:
        """Get label(s) for a sample based on task type."""
        if self.task == "multilabel":
            labels = self.df_labels.loc[sample_id, self.target_columns].values.astype(np.float32)
            return torch.tensor(labels, dtype=torch.float32)
        else:
            label = self.df_labels.loc[sample_id, self.target_columns[0]]
            return torch.tensor(label, dtype=torch.long)
    
    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        label = self._get_label(sample_id)
        filename = self.id_filename_map[sample_id]
        
        feat_embeds = []
        for encoder in self.slice_encoder_models:
            feat_file = os.path.join(self.feat_paths[encoder], filename)
            feat, _ = self._load_feat(feat_file)
            feat_embeds.append(feat)
        
        return {"feature_embeds": feat_embeds, "label": label, "sample_id": sample_id}

#TODO: multi-view dataset for linear probing
class MultiViewFeatClassificationDataset(Dataset):
    """
    Dataset for multi-view downstream classification tasks.
    Loads precomputed features from multiple view planes (axial, sagittal, coronal)
    for each exam, similar to get_pat_embs in COBRA for multi-slide patients.
    """
    pass