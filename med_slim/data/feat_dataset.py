import numpy as np
import pandas as pd
import torch
import random
import os
import logging
from typing import List, Tuple, Dict, Optional, Union
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
    Custom collate function for FeatClassificationDataset.
    
    Handles:
    (1) list of variable-length feature embedding dimensions from different slice encoders.
    (2) variable number of slice sequences in the batch: 
        adds zero-padding to the sequences to the max sequence length in the batch.
    
    Returns:
        dict with keys:
            - feature_embeds: List of K tensors, each [B, max_seq_len, embed_dim]
            - labels: [B] tensor of labels
            - sample_ids: List of sample IDs
            - seq_lengths: [B] tensor of actual sequence lengths for masking
    """
    labels = torch.stack([item["label"] for item in batch])
    sample_ids = [item["sample_id"] for item in batch]
    seq_lengths = torch.tensor([item["seq_length"] for item in batch], dtype=torch.long)
    
    all_feats_list = [item["feature_embeds"] for item in batch]
    K = len(all_feats_list[0])  # Number of slice encoders
    max_seq_len = max(seq_lengths).item()
    batch_size = len(batch)
    
    # Create a list to store the K collated feature batches
    features_list = []
    for k in range(K):
        # Gather k-th slice encoder features from all samples in the batch
        kth_feats = [sample_feats[k] for sample_feats in all_feats_list]
        embed_dim = kth_feats[0].shape[-1]
        
        # Pad each to max_seq_len
        padded = torch.zeros(batch_size, max_seq_len, embed_dim)
        for i, feat in enumerate(kth_feats):
            seq_len = feat.shape[0]
            padded[i, :seq_len, :] = feat
        
        features_list.append(padded)
    
    return {
        "features": features_list, 
        "seq_lengths": seq_lengths,
        "labels": labels,
        "sample_ids": sample_ids,
    }

def multiview_collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for MultiViewFeatClassificationDataset.
    
    Returns:
        Dict with keys:
            - feature_embeds: {view_plane: List of K tensors [B, max_seq_len, embed_dim]}
            - seq_length: {view_plane: tensor [B]}
            - labels: tensor [B] or [B, num_labels]
            - sample_ids: list of sample IDs
    """
    view_planes = list(batch[0]["feature_embeds"].keys())
    batch_size = len(batch)
    
    collated_features = {}
    collated_seq_lengths = {}
    
    for view_plane in view_planes:
        # Get all slice encoder features for this view plane
        all_feats_list = [item["feature_embeds"][view_plane] for item in batch]
        all_seq_lengths = [item["seq_length"][view_plane] for item in batch]
        
        K = len(all_feats_list[0])  # Number of slice encoders
        max_seq_len = max(all_seq_lengths)
        
        # Create list of K tensors for this view, each [B, max_seq_len, encoder_embed_dim]
        features_list = []
        for k in range(K):
            kth_feats = [sample_feats[k] for sample_feats in all_feats_list]
            embed_dim = kth_feats[0].shape[-1]
            
            # Pad each to max_seq_len
            padded = torch.zeros(batch_size, max_seq_len, embed_dim)
            for i, feat in enumerate(kth_feats):
                seq_len = feat.shape[0]
                padded[i, :seq_len, :] = feat
            
            features_list.append(padded)
        
        collated_features[view_plane] = features_list
        collated_seq_lengths[view_plane] = torch.tensor(all_seq_lengths, dtype=torch.long)
    
    # Collate labels
    labels = torch.stack([item["label"] for item in batch])
    sample_ids = [item["sample_id"] for item in batch]
    
    return {
        "features": collated_features,
        "seq_lengths": collated_seq_lengths,
        "labels": labels,
        "sample_ids": sample_ids,
    }

class PrecomputedFeatPairDataset(Dataset):
    """
    Dataset of constructing precomputed feature positivepairs at the exam level, which are used for contrastive learning.
    For multi-plane views, the concept of "same-plane positives" is implemented:
    when forming a pair, enforce both views to be from the same plane; use all planes across the dataset,
    but don't mixed within a pair.
    
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
    
    def _select_window_indices(self, num_slices: int, n_seq: int = 32) -> slice:
        """
        Sample the window of slices to keep, which is used for padding or subsampling the number of slices to n_seq.
        Return a slice object describing which z-indices to keep.
        """
        if num_slices <= n_seq:
            # keep all slices
            return slice(0, num_slices)
        else:
            # uniform sampling along z
            # pick start ~ Uniform(0, num_slices - n_seq), then take slices[start: start+n_seq]
            start_idx = np.random.randint(0, num_slices - n_seq + 1)
            return slice(start_idx, start_idx + n_seq)

    def _pad_feature_dim(self, x: torch.Tensor, n_seq: int = 32) -> torch.Tensor:
        """
        Pad the embedding dimension to the largest embedding dimension in the batch by padding with zeros if it is smaller than the target dimension
        """
        seq_len, embed_dim = x.shape
        
        if embed_dim < self.max_feature_dim:
            pad_size = self.max_feature_dim - embed_dim
            x = torch.cat([x, torch.zeros(seq_len, pad_size, device=x.device)], dim=1)
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
        assert metadata1["plane"] == metadata2["plane"], \
            f"Expected plane to be equal, but got {metadata1['plane']} and {metadata2['plane']}!"
        assert feats1.shape[0] == feats2.shape[0], \
            f"Expected number of slices to be equal, but got {feats1.shape[0]} and {feats2.shape[0]} for study {study_id} and view plane {selected_view_plane}!"
        # Pad or sample the number of slices to n_seq  and zero-pad number of features to the largest embedding dimension 
        # in order to let DataLoader collate function concatenate the tensors of the same shape per batch
        num_slices = feats1.shape[0]
        n_seq = 32
        with torch.no_grad():
            # choose a shared window indices
            window_slice = self._select_window_indices(num_slices, n_seq = n_seq)
            # crop both FMs using the SAME window, preserving order
            feats1 = feats1[window_slice]  # [S, D1], where S <= 32
            feats2 = feats2[window_slice]  # [S, D2]
            # zero-padding at the end of the sequence if S < n_seq
            S = feats1.shape[0]
            if S < n_seq:
                pad_size = n_seq - S
                # pad in slice dimension with zeros in feature space
                feats1 = torch.cat([feats1, torch.zeros(pad_size, feats1.shape[1], device=feats1.device)], dim=0)
                feats2 = torch.cat([feats2, torch.zeros(pad_size, feats2.shape[1], device=feats2.device)], dim=0)
            # pad feature dimension to max_feature_dim
            feats1, orig_embed_dim1 = self._pad_feature_dim(feats1.clone().detach())
            feats2, orig_embed_dim2 = self._pad_feature_dim(feats2.clone().detach()) 
        assert feats1.shape == feats2.shape, f"Expected shapes to be equal, but got {feats1.shape} and {feats2.shape}!"
        return feats1, torch.as_tensor(orig_embed_dim1), torch.as_tensor(S), feats2, torch.as_tensor(orig_embed_dim2), torch.as_tensor(S)

class PrecomputedFeatSupConDataset(Dataset):
    """
    Dataset of constructing precomputed feature positivepairs at the exam level, which are used for contrastive learning.
    For multi-plane views, the concept of "mixed-plane positives" is implemented:
    use SupCon loss (multi-positive loss) so that the loss function will treat every every view of the same exam (axial, sagittal, coronal) as separate positives in the denominator,
    explicitly tells the model: 'all these are positives for this exam'.
    """
    #TODO
    pass



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
        seq_lengths = []
        
        for encoder in self.slice_encoder_models:
            feat_file = os.path.join(self.feat_paths[encoder], filename)
            feat, _ = self._load_feat(feat_file)
            seq_lengths.append(feat.shape[0])
            feat_embeds.append(feat)
        
        # Assert all encoders have same sequence length in the same exam (same 3D volume)
        if len(seq_lengths) > 1:
            assert all(seq_len == seq_lengths[0] for seq_len in seq_lengths), (
                f"Sequence length mismatch across slice encoders for sample {sample_id}: "
                f"{dict(zip(self.slice_encoder_models, seq_lengths))}. "
                f"All encoders should have the same number of slices for the same exam."
            )
        
        return {
            "feature_embeds": feat_embeds,
            "seq_length": seq_lengths[0],
            "label": label,
            "sample_id": sample_id
        }


class MultiViewFeatClassificationDataset(Dataset):
    """
    Dataset for multi-view classification using precomputed slice features.
    Returns features from all view planes for each sample to enable end-to-end
    multi-view training.
    """
    def __init__(self,
                 feat_dir: str,
                 slice_encoder_models: List[str],
                 view_planes: List[str],
                 split: str,
                 annotations_path: str,
                 task: str,
                 target_columns: List[str]):
        """
        Args:
            feat_dir: Directory containing precomputed features
            slice_encoder_models: List of slice encoder model names
            view_planes: List of view planes (e.g., ["axial", "sagittal", "coronal"])
            split: Data split ("train" or "test")
            annotations_path: Path to the annotation CSV file
            task: Classification task type ("binary", "multiclass", or "multilabel")
            target_columns: List of column names to use as classification targets
        """
        if task not in ["binary", "multiclass", "multilabel"]:
            raise ValueError(f"`task` must be 'binary', 'multiclass', or 'multilabel', got '{task}'")
        if task in ["binary", "multiclass"] and len(target_columns) != 1:
            raise ValueError(f"For {task} classification, `target_columns` must have exactly 1 element")
        
        self.task = task
        self.target_columns = target_columns
        self.feat_dir = feat_dir
        self.slice_encoder_models = slice_encoder_models
        self.view_planes = view_planes
        self.split = split
        
        # Load labels
        self.df_labels = pd.read_csv(annotations_path)
        self.df_labels.set_index("ID", inplace=True)
        self.sample_ids = list(self.df_labels.index.astype(int))
        
        # Build feature paths for all encoders and views
        self.feat_paths = {}  # {view_plane: {model_name: path}}
        self.id_filename_map = {}
        
        for view_plane in self.view_planes:
            self.feat_paths[view_plane] = {}
            for model_name in self.slice_encoder_models:
                feat_path = os.path.join(feat_dir, model_name, split, view_plane)
                if not os.path.exists(feat_path):
                    raise FileNotFoundError(f"Feature path {feat_path} does not exist!")
                self.feat_paths[view_plane][model_name] = feat_path
                
                # Build id->filename mapping from first encoder/view
                if not self.id_filename_map:
                    feat_files = glob(os.path.join(feat_path, '*.safetensors'))
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
        
        # Load features for all views
        features_by_view = {}  # {view_plane: [encoder_feats]}
        seq_lengths_by_view = {}
        
        for view_plane in self.view_planes:
            feat_embeds = []
            seq_lengths = []
            
            for encoder in self.slice_encoder_models:
                feat_file = os.path.join(self.feat_paths[view_plane][encoder], filename)
                feat, _ = self._load_feat(feat_file)
                seq_lengths.append(feat.shape[0])
                feat_embeds.append(feat)
            
            # Assert all encoders have same sequence length for this view
            if len(seq_lengths) > 1:
                assert all(seq_len == seq_lengths[0] for seq_len in seq_lengths), (
                    f"Sequence length mismatch across slice encoders for sample {sample_id}, view {view_plane}"
                )
            
            features_by_view[view_plane] = feat_embeds
            seq_lengths_by_view[view_plane] = seq_lengths[0]
        
        return {
            "feature_embeds": features_by_view,  
            "seq_length": seq_lengths_by_view,  
            "label": label,
            "sample_id": sample_id,
        }
