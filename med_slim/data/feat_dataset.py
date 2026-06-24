import re
import numpy as np
import pandas as pd
import torch
import random
import os
import logging
from typing import List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import Dataset
from glob import glob
from tqdm import tqdm
from safetensors import safe_open
from collections import defaultdict
from med_slim.logging.setup import init_logging
init_logging()
logger = logging.getLogger(__name__)


class FeatureCache:
    """
    In-memory cache for safetensors feature files.

    Preloads all given paths at construction using multi-threaded I/O,
    then serves tensors and metadata from memory.
    Paths not in the cache are loaded from disk.
    """

    def __init__(self, paths):
        self._cache: Dict[str, Tuple[torch.Tensor, dict]] = {}
        self._preload(paths)

    @staticmethod
    def _read_tensor(path: str) -> Tuple[str, torch.Tensor, dict]:
        with safe_open(path, framework="pt", device="cpu") as f:
            return path, f.get_tensor("feats").clone(), f.metadata()

    def _preload(self, paths) -> None:
        paths = list(paths)
        logger.info(f"Preloading {len(paths)} feature files into memory "
                     f"(multi-threaded I/O)...")
        total_bytes = 0
        with ThreadPoolExecutor(max_workers=min(32, os.cpu_count() or 8)) as pool:
            for path, tensor, metadata in tqdm(
                pool.map(self._read_tensor, paths),
                total=len(paths),
                desc="Caching features in RAM",
            ):
                self._cache[path] = (tensor, metadata)
                total_bytes += tensor.nelement() * tensor.element_size()
        logger.info(f"Feature cache ready: {len(self._cache)} tensors, "
                     f"{total_bytes / 1e9:.2f} GB in RAM")

    def get(self, path: str) -> Tuple[torch.Tensor, dict]:
        """Return cached tensor and metadata, falling back to disk if not in cache."""
        cached = self._cache.get(path)
        if cached is not None:
            return cached
        with safe_open(path, framework="pt", device="cpu") as f:
            return f.get_tensor("feats"), f.metadata()


def _validate_feature_metadata(
    feats: torch.Tensor,
    metadata: dict,
) -> None:
    """
    Validate safetensor tiled metadata against the model/dataset config.

    Raises ``ValueError`` when global-only and tiled caches are mixed or when metadata
    disagrees with the tensor shape.
    """
    meta_regional = metadata.get("regional_tokens")
    meta_num_regions = metadata.get("num_regions")

    if feats.ndim == 2:
        if meta_regional is not None and int(meta_regional) > 0:
            raise ValueError(
                f"Loaded feature metadata regional_tokens={meta_regional} but "
                f"feats are global-only with shape {tuple(feats.shape)}."
            )
            
    elif feats.ndim == 3:
        num_regions = feats.shape[1]
        if meta_num_regions is not None and int(meta_num_regions) != num_regions:
            raise ValueError(
                f"num_regions metadata ({meta_num_regions}) != feats.shape[1] ({num_regions})."
            )
        if meta_regional is not None and num_regions != 1 + int(meta_regional):
            raise ValueError(
                f"regional_tokens metadata ({meta_regional}) implies "
                f"{1 + int(meta_regional)} regions, but feats have "
                f"{num_regions}."
            )

    else:
        raise ValueError(
            f"Loaded feature must have rank 2 or 3, got ndim={feats.ndim}."
        )    
    
def _read_and_validate_feat(
    feat_path: str,
) -> Tuple[torch.Tensor, dict]:
    """Load a safetensors feature file and validate its metadata."""
    with safe_open(feat_path, framework="pt", device="cpu") as f:
        feat = f.get_tensor("feats")
        metadata = f.metadata()
    _validate_feature_metadata(
        feat,
        metadata,
    )
    return feat, metadata


def ssl_packed_collate_fn(batch):
    """
    Packed sequence collate function for PrecomputedFeatPairDataset.
    
    Concatenates variable-length sequences and uses cumulative sequence lengths
    (cu_seqlens) to track boundaries to avoid padding waste. 
    Uses ``seq_len`` from each item to pack only real slices.
    
    Designed for use with ``PrecomputedFeatPairDataset(use_packed=True)``, which
    returns raw variable-length sequences without subsampling or zero-padding.
    
    Both views in a positive pair always share the same sequence length (same exam /
    plane / MRI sequence), so cu_seqlens and seq_idx are identical for both views.
    
    Efficient for:
    - Mamba2: Uses seq_idx to avoid passing states across sequence boundaries
    - Transformer with FlashAttention/varlen_attn: Uses cu_seqlens for variable-length attention
    
    Returns:
        dict with keys:
            - feats1, feats2: packed global-only features
              [total_slices, max_feature_dim] or tiled features
              [total_slices, num_tiled_regions, max_feature_dim]
            - cu_seqlens1, cu_seqlens2: cumulative sequence lengths [B+1] 
            - max_seqlen1, max_seqlen2: max real sequence length in batch
            - seq_idx1, seq_idx2: document index per token [total_slices]
            - orig_embed_dim1, orig_embed_dim2: original embedding dimensions [B]
            - label (optional): stacked labels [B, C] when annotations exist
            - has_label (optional): boolean tensor [B] when annotations exist
    """
    batch_size = len(batch)
    orig_embed_dims1 = torch.stack([item["orig_embed_dim1"].to(dtype=torch.long) for item in batch], dim=0)
    orig_embed_dims2 = torch.stack([item["orig_embed_dim2"].to(dtype=torch.long) for item in batch], dim=0)
    
    feats1_list = [item["feats1"] for item in batch]
    feats2_list = [item["feats2"] for item in batch]
    seq_lens = torch.tensor([f.shape[0] for f in feats1_list], dtype=torch.int32)
    
    assert all(f1.shape[0] == f2.shape[0] for f1, f2 in zip(feats1_list, feats2_list)), (
        "feats1 and feats2 must have the same sequence length per sample "
        "(positive pairs come from the same MRI exam/plane/sequence)."
    )
    
    # Compute cumulative sequence lengths (cu_seqlens)
    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32)
    cu_seqlens[1:] = torch.cumsum(seq_lens, dim=0)
    
    assert all(f.ndim == feats1_list[0].ndim for f in feats1_list + feats2_list), (
        "Packed batches cannot mix global-only CLS features [num_slices, max_feature_dim] "
        "with tiled multi-crop CLS features [num_slices, num_tiled_regions, max_feature_dim]."
    )
    if feats1_list[0].ndim == 3:
        num_tiled_regions = feats1_list[0].shape[1]
        assert all(f.shape[1] == num_tiled_regions for f in feats1_list + feats2_list), (
            "All tiled packed features in a batch must have the same num_tiled_regions."
        )

    # Pack features: concatenate variable-length sequences along the slice axis.
    feats1_packed = torch.cat(feats1_list, dim=0)  # [total_slices, (num_tiled_regions,) max_feature_dim]
    feats2_packed = torch.cat(feats2_list, dim=0)  # [total_slices, (num_tiled_regions,) max_feature_dim]
    
    # Vectorized building seq_idx: repeat each batch index by its sequence length
    seq_idx = torch.repeat_interleave(torch.arange(batch_size, dtype=torch.int32), seq_lens)
    
    phys_pos_list = [
        item.get(
            "physical_positions",
            PrecomputedFeatPairDataset._compute_physical_positions(
                np.arange(item["feats1"].shape[0]),
                item["feats1"].shape[0],
                item["feats1"].shape[0],
            ),
        )
        for item in batch
    ]
    phys_pos_packed = torch.cat(phys_pos_list, dim=0)  # [total_slices]
    
    result = {
        "feats1": feats1_packed,   # [total_slices, (num_tiled_regions,) max_feature_dim]
        "feats2": feats2_packed,
        "cu_seqlens1": cu_seqlens,  # [B+1] — shared across both views
        "cu_seqlens2": cu_seqlens,
        "max_seqlen1": seq_lens.max().item(),
        "max_seqlen2": seq_lens.max().item(),
        "seq_idx1": seq_idx,        # [total_slices]
        "seq_idx2": seq_idx,
        "orig_embed_dim1": orig_embed_dims1,  # [B]
        "orig_embed_dim2": orig_embed_dims2,
        "batch_size": batch_size,
        "physical_positions": phys_pos_packed,  # [total_slices]
    }
    
    # Forward labels for semi-supervised contrastive learning
    if "label" in batch[0]:
        result["label"] = torch.stack([item["label"] for item in batch], dim=0)
        result["has_label"] = torch.stack([item["has_label"] for item in batch], dim=0)
    
    return result


def linear_classifier_collate_fn(batch):
    """
    Custom collate function for FeatClassificationDataset.
    
    Handles:
    (1) list of variable-length feature embedding dimensions from different slice encoders.
    (2) variable number of slice sequences in the batch: 
        adds zero-padding to the sequences to the max sequence length in the batch.
    
    Returns:
        dict with keys:
            - features: List of K tensors, each [B, max_seq_len, embed_dim] for global-only or [B, max_seq_len, num_tiled_regions, embed_dim] for tiled multi-crop CLS.
            - labels: [B] tensor of labels
            - sample_ids: List of sample IDs
            - seq_lengths: [B] tensor of actual sequence lengths for masking
    """
    labels = torch.stack([item["label"] for item in batch])
    sample_ids = [item["sample_id"] for item in batch]
    seq_lengths = torch.tensor([item["seq_length"] for item in batch], dtype=torch.long)
    physical_positions = [item["physical_positions"] for item in batch]
    
    all_feats_list = [item["feature_embeds"] for item in batch]
    K = len(all_feats_list[0])  # Number of slice encoders
    max_seq_len = max(feat[0].shape[0] for feat in all_feats_list)
    batch_size = len(batch)
    
    # Create a list to store the K collated feature batches
    features_list = []
    for k in range(K):
        # Gather k-th slice encoder features from all samples in the batch
        kth_feats = [sample_feats[k] for sample_feats in all_feats_list]
        trailing_dims = kth_feats[0].shape[1:] # (embed_dim,) for global-only or (num_tiled_regions, embed_dim) for tiled multi-crop CLS
        
        # Pad each to max_seq_len
        padded = torch.zeros(batch_size, max_seq_len, *trailing_dims)
        for i, feat in enumerate(kth_feats):
            seq_len = feat.shape[0]
            padded[i, :seq_len] = feat
        
        features_list.append(padded)
    
    physical_positions_padded = torch.zeros(batch_size, max_seq_len, dtype=torch.float32)
    for i, pos in enumerate(physical_positions):
        seq_len = pos.shape[0]
        physical_positions_padded[i, :seq_len] = pos

    return {
        "features": features_list, 
        "seq_lengths": seq_lengths,
        "physical_positions": physical_positions_padded,
        "labels": labels,
        "sample_ids": sample_ids,
    }

def multiview_classifier_collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for MultiViewFeatClassificationDataset.
    
    Returns:
        Dict with keys:
            - features: {view_plane: List of K tensors [B, max_seq_len, embed_dim] for global-only or [B, max_seq_len, num_tiled_regions, embed_dim] for tiled multi-crop CLS}
            - seq_length: {view_plane: tensor [B]}
            - labels: tensor [B] or [B, num_labels]
            - sample_ids: list of sample IDs
    """
    view_planes = list(batch[0]["feature_embeds"].keys())
    batch_size = len(batch)
    
    collated_features = {}
    collated_seq_lengths = {}
    collated_physical_positions = {}
    
    for view_plane in view_planes:
        # Get all slice encoder features for this view plane
        all_feats_list = [item["feature_embeds"][view_plane] for item in batch]
        all_seq_lengths = [item["seq_length"][view_plane] for item in batch]
        all_physical_positions = [item["physical_positions"][view_plane] for item in batch]
        
        K = len(all_feats_list[0])  # Number of slice encoders
        max_seq_len = max(feat[0].shape[0] for feat in all_feats_list)
        
        # Create list of K tensors for this view, each [B, max_seq_len, embed_dim] for global-only 
        # or [B, max_seq_len, num_tiled_regions, embed_dim] for tiled multi-crop CLS.
        features_list = []
        for k in range(K):
            kth_feats = [sample_feats[k] for sample_feats in all_feats_list]
            trailing_dims = kth_feats[0].shape[1:]
            
            # Pad each to max_seq_len
            padded = torch.zeros(batch_size, max_seq_len, *trailing_dims)
            for i, feat in enumerate(kth_feats):
                seq_len = feat.shape[0]
                padded[i, :seq_len] = feat
            
            features_list.append(padded)
        
        collated_features[view_plane] = features_list
        collated_seq_lengths[view_plane] = torch.tensor(all_seq_lengths, dtype=torch.long)
        physical_positions_padded = torch.zeros(batch_size, max_seq_len, dtype=torch.float32)
        for i, pos in enumerate(all_physical_positions):
            seq_len = pos.shape[0]
            physical_positions_padded[i, :seq_len] = pos
        collated_physical_positions[view_plane] = physical_positions_padded
    
    # Collate labels
    labels = torch.stack([item["label"] for item in batch])
    sample_ids = [item["sample_id"] for item in batch]
    
    return {
        "features": collated_features,
        "seq_lengths": collated_seq_lengths,
        "physical_positions": collated_physical_positions,
        "labels": labels,
        "sample_ids": sample_ids,
    }

class PrecomputedFeatPairDataset(Dataset):
    """
    Dataset for constructing cross-FM positive pairs at the patient-study level for contrastive learning.
    
    Entries are grouped by patient study. 
    For datasets with multiple MRI sequences per view plane, 
    all MRI sequences from the same patient study are grouped together.
    Each ``__getitem__`` call stochastically selects a plane and a series (UID),
    then forms a cross-FM positive pair from that series.
    
    SSL setting:
        Positive pairs are constructed from different foundation models applied to the same exam and view plane.
    
    Semi-supervised learning setting:
        When dataset configs include ``annotations_path`` and ``target_columns``, classification
        labels are loaded and returned alongside features. 
        This enables semi-supervised contrastive learning (SemiSupCon)
        where labeled samples contribute additional same-class positives via the SupCon loss, 
        while unlabeled samples use standard InfoNCE.

    To handle variable-length slices, we sample or pad to a fixed number of slices ``num_target_slices``:
        - Volumes with more slices than ``num_target_slices`` are uniformly subsampled (preserving anatomical order).
        - Volumes with fewer slices than ``num_target_slices`` are zero-padded.
    
    Directory structure:
        {feat_dir}/{model}/{split}/{plane}/*.safetensors
    
    Data structure:
        feat_path_dict[study_id][plane] = [uid1, uid2, ...]    # one UID per MRI series
        feat_dir_map[study_id] = feat_dir                      # base directory for path reconstruction
    """
    def __init__(self, 
                 feat_dirs: List[Dict[str, str]], 
                 slice_encoder_models: List[str], 
                 view_planes: List[str], 
                 split: str, 
                 max_feature_dim: int,
                 num_target_slices: int = 32,
                 cache_in_memory: bool = False,
                 use_packed: bool = False,
                 ssl_fm_mode: str = "pair",
                 fm_subset_size: int | None = None,
                 fm_subset_min_overlap: int = 1,
                 fm_subset_max_overlap: int | None = None,
                 fm_name_to_id: Dict[str, int] | None = None,
                 return_fm_names: bool = False):
        """
        Args:
            feat_dirs: List of dataset config dictionaries. Each dict must have:
                - "name" (str): Dataset name (e.g., "mrnet")
                - "feat_dir" (str): Path to precomputed features
                And optionally for semi-supervised learning:
                - "annotations_path" (str): Path to annotation CSV with columns [ID, target1, target2, ...]
                - "target_columns" (list[str]): Column names to use as classification targets
                                                (e.g., ["abnormal", "acl", "meniscus"])
            slice_encoder_models: List of slice encoder model names (e.g., ["dinov2", "rad-dino"])
            view_planes: List of view planes (e.g., ["axial", "sagittal", "coronal"])
            split: Data split ("train" or "val")
            max_feature_dim: Maximum feature dimension for padding
            num_target_slices: Target number of slices for all outputs. Volumes with more slices 
                               are uniformly subsampled (order-preserving); volumes with fewer are 
                               zero-padded. Default: 32. Ignored when use_packed=True.
            cache_in_memory: If True, preload all feature tensors into RAM at init to eliminate per-epoch disk I/O.
            use_packed: If True, return raw variable-length sequences (no subsampling or
                        zero-padding). Must be used with ssl_packed_collate_fn in the DataLoader.
            ssl_fm_mode: Positive-pair construction strategy.
                - "pair" (default): cross-FM pair baseline. Each view is exactly one FM;
                  preserves the original outputs/shapes exactly.
                - "subset": each view is a controlled-overlap subset of FMs. The main
                  router SSL mode. Sampled sets always have fixed size fm_subset_size.
            fm_subset_size: Number of FMs per view in subset mode. Required for "subset".
                Must be strictly less than len(slice_encoder_models) so two non-identical
                subsets can always be formed.
            fm_subset_min_overlap: Minimum shared FMs between the two views (0..fm_subset_size).
            fm_subset_max_overlap: Maximum shared FMs between the two views. Defaults to
                fm_subset_size - 1 so views are never forced to be identical.
            fm_name_to_id: Optional mapping FM name -> stable global FM ID. When None, IDs are
                derived from slice_encoder_models order. FM identity never comes from feature dim.
            return_fm_names: If True, also return the selected FM names per view (debugging/logging).
        """
        self.feat_dirs = feat_dirs
        self.slice_encoder_models = slice_encoder_models
        self.view_planes = view_planes
        self.split = split
        self.max_feature_dim = max_feature_dim
        self.num_target_slices = num_target_slices
        self.use_packed = use_packed

        # FM subset SSL configuration
        if ssl_fm_mode not in ("pair", "subset"):
            raise ValueError(
                f"Invalid ssl_fm_mode '{ssl_fm_mode}'. Must be 'pair' or 'subset'."
            )
        self.ssl_fm_mode = ssl_fm_mode
        self.fm_subset_size = fm_subset_size
        self.fm_subset_min_overlap = fm_subset_min_overlap
        self.fm_subset_max_overlap = fm_subset_max_overlap
        self.return_fm_names = return_fm_names

        # Stable global FM IDs follow slice_encoder_models order unless overridden.
        if fm_name_to_id is None:
            fm_name_to_id = {name: i for i, name in enumerate(slice_encoder_models)}
        else:
            missing = [m for m in slice_encoder_models if m not in fm_name_to_id]
            if missing:
                raise ValueError(f"fm_name_to_id is missing IDs for FMs: {missing}.")
        self.fm_name_to_id = fm_name_to_id
        self.num_fms = len(slice_encoder_models)

        if ssl_fm_mode == "subset":
            if use_packed:
                raise NotImplementedError(
                    "Packed subset FM mode is not supported yet. Use padded subset batches "
                    "(use_packed=False) for fm_pooling router SSL."
                )
            if fm_subset_size is None:
                raise ValueError("fm_subset_size is required when ssl_fm_mode='subset'.")
            if not (1 <= fm_subset_size < self.num_fms):
                raise ValueError(
                    f"fm_subset_size must satisfy 1 <= size < num_fms ({self.num_fms}); "
                    f"got {fm_subset_size}. A strict subset is required so two non-identical "
                    "views can always be sampled (all-FM is only ever used on one side)."
                )
            if not (0 <= fm_subset_min_overlap <= fm_subset_size):
                raise ValueError(
                    f"fm_subset_min_overlap must satisfy 0 <= overlap <= fm_subset_size "
                    f"({fm_subset_size}); got {fm_subset_min_overlap}."
                )
            fm_max_overlap = (
                fm_subset_max_overlap
                if fm_subset_max_overlap is not None
                else fm_subset_size - 1
            )
            # Cap at fm_subset_size - 1 so the two views can never be forced identical.
            fm_max_overlap = min(fm_max_overlap, fm_subset_size - 1)
            if fm_max_overlap < fm_subset_min_overlap:
                raise ValueError(
                    f"fm_subset_max_overlap ({fm_max_overlap}) < fm_subset_min_overlap "
                    f"({fm_subset_min_overlap}) after capping at fm_subset_size - 1."
                )
            min_possible_overlap = max(0, 2 * fm_subset_size - self.num_fms)
            fm_min_overlap = max(fm_subset_min_overlap, min_possible_overlap)
            if fm_min_overlap > fm_max_overlap:
                raise ValueError(
                    "No valid non-identical FM subset pair exists with the requested overlap "
                    f"constraints. num_fms={self.num_fms}, fm_subset_size={fm_subset_size}, "
                    f"min_overlap={fm_subset_min_overlap}, max_overlap={fm_max_overlap}, "
                    f"minimum possible overlap={min_possible_overlap}."
                )
            self._fm_min_overlap = fm_min_overlap
            self._fm_max_overlap = fm_max_overlap

        self.feat_dir_map = {}
        self.feat_path_dict = self._get_feat_path_dict_by_study_id()
        self.study_ids = list(self.feat_path_dict.keys())
        self._feat_cache = None
        if cache_in_memory:
            unique_paths = set()
            for study_id in self.study_ids:
                for plane, uid_list in self.feat_path_dict[study_id].items():
                    for uid in uid_list:
                        for model in self.slice_encoder_models:
                            unique_paths.add(self._get_feat_path(study_id, model, plane, uid))
            self._feat_cache = FeatureCache(unique_paths)
        
        # Load annotations for semi-supervised contrastive learning
        self._load_annotations(feat_dirs)
    
    @staticmethod
    def _extract_exam_id(uid: str) -> str:
        """Extract patient exam ID from UID.
        
        Groups multiple MRI series from the same patient study together.
        
        For multi-series datasets, the UID follows the format: '{patient_id}_{series_name}'
            e.g., 'patient_123_MR4_0b2ac15c' → 'patient_123'
        For single-series datasets, the UID is simply the patient ID.
            e.g., 'patient_123' → 'patient_123'

        Returns:
            The patient exam ID from the UID.
        """
        mr_match = re.search(r'_MR\d+', uid)
        if mr_match:
            return uid[:mr_match.start()]
        return uid
        
    def _get_feat_path_dict_by_study_id(self):
        """
        Build a dictionary mapping study_id -> view_plane -> list of MRI sequence UIDs.
        
        Returns:
            Dict[str, Dict[str, List[str]]]:
                study_id -> plane -> [uid1, uid2, ...]
        """
        logger.info(f'Selected slice encoder models: {self.slice_encoder_models}')
        logger.info(f'Selected view planes: {self.view_planes}')
        
        ref_fm = self.slice_encoder_models[0]
        feat_path_dict = defaultdict(lambda: defaultdict(list))
        
        for dataset in tqdm(self.feat_dirs, desc="Loading precomputed feature datasets...", leave=False):
            dataset_name = dataset["name"]
            feat_dir = dataset["feat_dir"]
            
            for view_plane in self.view_planes:
                feat_path = os.path.join(feat_dir, ref_fm, self.split, view_plane)
                
                if not os.path.exists(feat_path):
                    raise FileNotFoundError(f"Feature path {feat_path} does not exist!")
                
                feat_files = glob(os.path.join(feat_path, "*.safetensors"))
                assert len(feat_files) > 0, f"Couldn't find any feat files in path {feat_path}!"
                
                for feat_file in feat_files:
                    uid = os.path.basename(feat_file).split(".")[0]
                    exam_id = self._extract_exam_id(uid)
                    study_id = f"{dataset_name}_{exam_id}"
                    feat_path_dict[study_id][view_plane].append(uid)
                    self.feat_dir_map[study_id] = feat_dir
        
        # Validate all FMs have consistent feature files with the reference FM
        for dataset in self.feat_dirs:
            feat_dir = dataset["feat_dir"]
            for view_plane in self.view_planes:
                ref_path = os.path.join(feat_dir, ref_fm, self.split, view_plane)
                ref_uids = {os.path.basename(f).split(".")[0] 
                            for f in glob(os.path.join(ref_path, "*.safetensors"))}
                
                for fm in self.slice_encoder_models[1:]:
                    fm_path = os.path.join(feat_dir, fm, self.split, view_plane)
                    if not os.path.exists(fm_path):
                        raise FileNotFoundError(
                            f"Feature path {fm_path} does not exist! "
                            f"Make sure to precompute features for FM '{fm}'."
                        )
                    fm_uids = {os.path.basename(f).split(".")[0] 
                               for f in glob(os.path.join(fm_path, "*.safetensors"))}
                    missing = ref_uids - fm_uids
                    if missing:
                        raise FileNotFoundError(
                            f"FM '{fm}' is missing {len(missing)} feature files in {fm_path} "
                            f"that exist for reference FM '{ref_fm}'. "
                            f"Examples: {sorted(missing)[:5]}"
                        )
                    extra = fm_uids - ref_uids
                    if extra:
                        raise FileNotFoundError(
                            f"FM '{fm}' has {len(extra)} extra files in {fm_path} "
                            f"not found for reference FM '{ref_fm}'. "
                            f"Examples: {sorted(extra)[:5]}"
                        )
        
        # Compute statistics
        total_series = sum(
            len(uid_list)
            for planes in feat_path_dict.values()
            for uid_list in planes.values()
        )
        studies_with_multi_series = sum(
            1 for planes in feat_path_dict.values()
            if any(len(uid_list) > 1 for uid_list in planes.values())
        )
        
        logger.info(
            f"Found {len(feat_path_dict)} patient studies with {total_series} total series. "
            f"{studies_with_multi_series} studies have multiple series per plane."
        )

        return feat_path_dict
    
    def _get_feat_path(self, study_id: str, model_name: str, plane: str, uid: str) -> str:
        """Reconstruct the full feature file path from components."""
        return os.path.join(
            self.feat_dir_map[study_id], model_name, self.split, plane, f"{uid}.safetensors"
        )
        
    def _load_annotations(self, feat_dirs: List[Dict[str, str]]) -> None:
        """
        Load classification annotations for datasets that provide them.
        
        Enables semi-supervised contrastive learning where labeled samples contribute
        additional same-class positives to the contrastive loss (SupCon), while unlabeled
        samples use standard InfoNCE.
        
        Annotations are keyed by (dataset_name, exam_id) to match the study_id format
        used in the feature path dict: "{dataset_name}_{exam_id}".
        """
        self._annotations = {}  # {dataset_name: {exam_id_str: label_tensor}}
        self._num_labels = 0
        
        for dataset in feat_dirs:
            annotations_path = dataset.get("annotations_path")
            target_columns = dataset.get("target_columns")
            
            if annotations_path is None and target_columns is None:
                continue  # No annotations for this dataset (unlabeled)
            
            if annotations_path is None or target_columns is None:
                raise ValueError(
                    f"Dataset '{dataset['name']}': both 'annotations_path' and 'target_columns' "
                    f"must be provided together for semi-supervised learning."
                )
            
            if not os.path.exists(annotations_path):
                raise FileNotFoundError(
                    f"Annotations file not found: {annotations_path}"
                )
            
            dataset_name = dataset["name"]
            # Read with dtype=str for ID to preserve zero-padded IDs (e.g., "0001")
            df = pd.read_csv(annotations_path, dtype={"ID": str})
            df.set_index("ID", inplace=True)
            
            # Validate target columns
            missing = [c for c in target_columns if c not in df.columns]
            if missing:
                raise ValueError(
                    f"Target columns {missing} not found in {annotations_path}. "
                    f"Available columns: {list(df.columns)}"
                )
            
            self._num_labels = len(target_columns)
            self._annotations[dataset_name] = {}
            
            for exam_id in df.index:
                label_values = df.loc[exam_id, target_columns].values.astype(np.float32)
                self._annotations[dataset_name][str(exam_id)] = torch.tensor(label_values)
            
            logger.info(
                f"Loaded {len(self._annotations[dataset_name])} annotations for '{dataset_name}' "
                f"with {self._num_labels} target columns: {target_columns}"
            )
        
        if self._annotations:
            labeled_datasets = list(self._annotations.keys())
            logger.info(f"Semi-supervised mode: labeled dataset(s) = {labeled_datasets}")
        else:
            logger.info("Self-supervised mode: no annotations provided")

    @property
    def has_annotations(self) -> bool:
        """Whether any dataset has annotations for semi-supervised learning."""
        return self._num_labels > 0

    @property
    def num_labels(self) -> int:
        """Number of labels (0 if no annotations)."""
        return self._num_labels

    def _load_feats(self, feat_path: str) -> Tuple[torch.Tensor, dict]:
        with safe_open(feat_path, framework="pt", device="cpu") as f:
            feats = f.get_tensor("feats").clone()
            metadata = f.metadata()
        assert feats.ndim in (2, 3), (
            "Expected feature tensor with 2 ([num_slices, embed_dim]) or 3 "
            "([num_slices, num_tiled_regions, embed_dim]) dimensions, "
            f"but got {feats.ndim=}!"
        )
        assert metadata["plane"] in self.view_planes, f"Expected plane to be in {self.view_planes}, but got {metadata['plane']}!"
        assert metadata["model_name"] in self.slice_encoder_models, f"Expected model name to be in {self.slice_encoder_models}, but got {metadata['model_name']}!"
        _validate_feature_metadata(
            feats,
            metadata,
        )
        return feats, metadata
        
    @staticmethod
    def _compute_physical_positions(
        selected_indices: np.ndarray,
        num_slices: int,
        num_target_slices: int,
    ) -> torch.Tensor:
        """
        Compute normalized relative depth in [0, 1] for selected slice indices.

        Maps each subsampled index to its fractional position through the original
        volume (0 = first slice, 1 = last slice). This is spacing-independent and
        portable across datasets with different slice thickness or resampling grids.

        Padded positions (beyond ``len(selected_indices)``) are set to 0.

        Returns:
            Tensor [num_target_slices] of relative depth values in [0, 1].
        """
        denom = max(num_slices - 1, 1)
        positions = selected_indices.astype(np.float32) / denom

        pos_tensor = torch.zeros(num_target_slices, dtype=torch.float32)
        pos_tensor[: len(positions)] = torch.from_numpy(positions)
        return pos_tensor

    def _get_feats(self, feat_path: str) -> Tuple[torch.Tensor, dict]:
        """Return features and metadata from cache if available, otherwise load from disk."""
        if self._feat_cache is not None:
            return self._feat_cache.get(feat_path)
        return self._load_feats(feat_path)
        
    @staticmethod
    def _compute_subsample_indices(num_slices: int, num_target: int) -> np.ndarray:
        """
        Compute deterministic evenly-spaced indices for subsampling.

        Returns indices into the original [0, num_slices) range.  When
        ``num_slices <= num_target``, returns ``np.arange(num_slices)``
        (no subsampling).
        """
        if num_slices <= num_target:
            return np.arange(num_slices)
        return np.round(np.linspace(0, num_slices - 1, num_target)).astype(int)

    def _subsample_or_pad_slices(
        self, feats: torch.Tensor, indices: np.ndarray | None = None
    ) -> Tuple[torch.Tensor, int, np.ndarray]:
        """
        Subsample or zero-pad a slice feature tensor to a fixed number of slices.

        Uses deterministic evenly-spaced subsampling (linspace) to preserve
        anatomical coverage proportionally.  Both FM views in a positive pair
        share the same indices so that they see identical anatomy.

        Args:
            feats: Feature tensor [num_slices, embed_dim] (global-only CLS) or
                   [num_slices, num_tiled_regions, embed_dim] (tiled multi-crop CLS).
            indices: Pre-computed subsampling indices (optional).  
                     When None, computed via `_compute_subsample_indices`.

        Returns:
            Tuple of (features [num_target_slices, ...],
                      actual_num_real_slices,
                      selected_indices into original volume).
        """
        num_slices = feats.shape[0]

        if indices is None:
            indices = self._compute_subsample_indices(num_slices, self.num_target_slices)

        if num_slices <= self.num_target_slices:
            # Zero-pad to target length
            pad_size = self.num_target_slices - num_slices
            if feats.ndim == 2:          # [num_slices, embed_dim]
                pad = (0, 0, 0, pad_size)
            elif feats.ndim == 3:        # [num_slices, num_tiled_regions, embed_dim]
                pad = (0, 0, 0, 0, 0, pad_size)
            else:
                raise ValueError(f"Unsupported feature tensor rank {feats.ndim}.")
            padded = torch.nn.functional.pad(feats, pad, value=0.0)
            return padded, num_slices, indices
        else:
            return feats[indices], self.num_target_slices, indices

    def _pad_feature_dim(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Pad the embedding dimension to the largest embedding dimension in the batch by padding with zeros if it is smaller than the target dimension.
        """
        embed_dim = x.shape[-1]
        
        if embed_dim < self.max_feature_dim:
            pad_size = self.max_feature_dim - embed_dim
            x = torch.nn.functional.pad(x, (0, pad_size), mode='constant', value=0)
        return x, embed_dim
    
    def __len__(self):
        return len(self.study_ids)

    def _sample_fm_subsets(self) -> Tuple[List[int], List[int]]:
        """
        Sample two non-identical FM subsets (global FM IDs) with controlled overlap.

        Returns (view1_ids, view2_ids). Guarantees set(view1) != set(view2) and at least
        fm_subset_min_overlap shared FMs.
        """
        all_ids = list(range(self.num_fms))
        size = self.fm_subset_size

        # Directly sample a valid ordered pair with exact overlap. This avoids retry loops
        # and enforces the requested overlap range after accounting for finite num_fms.
        overlap = random.randint(self._fm_min_overlap, self._fm_max_overlap)
        view1 = set(random.sample(all_ids, size))
        shared = set(random.sample(list(view1), overlap))
        outside_view1 = [fid for fid in all_ids if fid not in view1]
        view2_extra = set(random.sample(outside_view1, size - overlap))
        view2 = shared | view2_extra

        return sorted(view1), sorted(view2)

    def _build_subset_view(
        self,
        fm_id_list: List[int],
        feats_by_id: Dict[int, torch.Tensor],
        indices: np.ndarray,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stack one subset-SSL view: [K_sub, num_slices, (regions,) max_feature_dim] + dims [K_sub]."""
        view_feats = []
        view_dims = []
        for fid in fm_id_list:
            feats, _, _ = self._subsample_or_pad_slices(feats_by_id[fid].clone(), indices)
            feats, orig_dim = self._pad_feature_dim(feats)
            view_feats.append(feats)
            view_dims.append(orig_dim)
        stacked = torch.stack(view_feats, dim=0)
        return stacked, torch.tensor(view_dims, dtype=torch.long)

    def _getitem_subset(self, study_id: str, plane: str, uid: str) -> Dict:
        """Build a subset-mode positive pair: each view is a controlled-overlap FM set."""
        fm_ids1, fm_ids2 = self._sample_fm_subsets()
        needed_ids = sorted(set(fm_ids1) | set(fm_ids2))

        feats_by_id: Dict[int, torch.Tensor] = {}
        metadata_by_id: Dict[int, dict] = {}
        num_slices = None
        ref_ndim = None
        ref_regions = None
        for fid in needed_ids:
            fm_name = self.slice_encoder_models[fid]
            feats, metadata = self._get_feats(self._get_feat_path(study_id, fm_name, plane, uid))
            feats_by_id[fid] = feats
            metadata_by_id[fid] = metadata
            if num_slices is None:
                num_slices = feats.shape[0]
                ref_ndim = feats.ndim
                ref_regions = feats.shape[1] if feats.ndim == 3 else None
            else:
                assert feats.shape[0] == num_slices, (
                    f"Slice-count mismatch across FMs for study {study_id}, plane {plane}: "
                    f"FM '{fm_name}' has {feats.shape[0]} slices vs {num_slices}."
                )
                assert feats.ndim == ref_ndim, (
                    f"Token layout mismatch across FMs for study {study_id}: FM '{fm_name}' "
                    f"has {feats.ndim}D features but expected {ref_ndim}D. Do not mix "
                    "global-only and tiled multi-crop CLS caches."
                )
                if ref_ndim == 3:
                    assert feats.shape[1] == ref_regions, (
                        f"num_tiled_regions mismatch across FMs for study {study_id}: FM "
                        f"'{fm_name}' has {feats.shape[1]} regions vs {ref_regions}."
                    )

        indices = self._compute_subsample_indices(num_slices, self.num_target_slices)
        physical_positions = self._compute_physical_positions(
            indices, num_slices, self.num_target_slices,
        )

        feats1, dims1 = self._build_subset_view(fm_ids1, feats_by_id, indices)
        feats2, dims2 = self._build_subset_view(fm_ids2, feats_by_id, indices)
        seq_len = min(num_slices, self.num_target_slices)

        result = {
            "feats1": feats1,                              # [K_sub, num_slices, (regions,) max_feature_dim]
            "feats2": feats2,
            "orig_embed_dim1": dims1,                      # [K_sub]
            "orig_embed_dim2": dims2,                      # [K_sub]
            "fm_ids1": torch.tensor(fm_ids1, dtype=torch.long),   # [K_sub]
            "fm_ids2": torch.tensor(fm_ids2, dtype=torch.long),   # [K_sub]
            "seq_len": torch.as_tensor(seq_len, dtype=torch.long),
            "physical_positions": physical_positions,      # [num_target_slices] in [0, 1]
            "study_id": study_id,
            "dataset_name": study_id.split("_", 1)[0],
            "plane": plane,
            "uid": uid,
        }

        if self.return_fm_names:
            result["fm_names1"] = [self.slice_encoder_models[i] for i in fm_ids1]
            result["fm_names2"] = [self.slice_encoder_models[i] for i in fm_ids2]

        self._attach_labels(result, study_id)
        return result

    def _attach_labels(self, result: Dict, study_id: str) -> None:
        """Attach semi-supervised classification labels to a result dict when available."""
        if self._num_labels <= 0:
            return
        dataset_name, exam_id = study_id.split("_", 1)
        label_tensor = self._annotations.get(dataset_name, {}).get(exam_id, None)
        if label_tensor is not None:
            result["label"] = label_tensor.clone()
            result["has_label"] = torch.tensor(True)
        else:
            result["label"] = torch.zeros(self._num_labels, dtype=torch.float32)
            result["has_label"] = torch.tensor(False)

    def __getitem__(self, idx):
        study_id = self.study_ids[idx]
        
        # Select a random view plane from the study
        available_planes = list(self.feat_path_dict[study_id].keys())
        selected_view_plane = random.choice(available_planes)
        
        # Select a random MRI sequence UID
        uid_list = self.feat_path_dict[study_id][selected_view_plane]
        selected_uid = random.choice(uid_list)

        # Subset mode: each view is a controlled-overlap FM set
        if self.ssl_fm_mode == "subset":
            return self._getitem_subset(study_id, selected_view_plane, selected_uid)

        # Cross-FM positive pair from the selected series
        fm1, fm2 = random.sample(self.slice_encoder_models, 2)
        feat_path1 = self._get_feat_path(study_id, fm1, selected_view_plane, selected_uid)
        feat_path2 = self._get_feat_path(study_id, fm2, selected_view_plane, selected_uid)
        feats1, metadata1 = self._get_feats(feat_path1)
        feats2, metadata2 = self._get_feats(feat_path2)
        
        num_slices = feats1.shape[0]
        assert num_slices == feats2.shape[0], (
            f"Expected same number of slices for positive pair (same patient with same view plane and MRI sequence), "
            f"but got {feats1.shape[0]} and {feats2.shape[0]} for study {study_id}, "
            f"plane {selected_view_plane}."
        )
        # Tiled multi-crop CLS: both views must expose the same num_tiled_regions
        assert feats1.ndim == feats2.ndim, (
            f"Token layout mismatch in positive pair for study {study_id}: "
            f"FM '{fm1}' has {feats1.ndim}D features but FM '{fm2}' has {feats2.ndim}D. "
        )
        if feats1.ndim == 3:
            assert feats1.shape[1] == feats2.shape[1], (
                f"num_tiled_regions mismatch in positive pair for study {study_id}: "
                f"{feats1.shape[1]} v.s. {feats2.shape[1]}."
            )

        if self.use_packed:
            seq_len = num_slices
            indices = np.arange(num_slices)
        else:
            # Compute shared indices once for both views (identical slices)
            indices = self._compute_subsample_indices(num_slices, self.num_target_slices)
            feats1, seq_len, indices = self._subsample_or_pad_slices(feats1, indices)
            feats2, _, _ = self._subsample_or_pad_slices(feats2, indices)
        
        # Relative depth in [0, 1] for sinusoidal PE
        physical_positions = self._compute_physical_positions(
            indices,
            num_slices,
            num_slices if self.use_packed else self.num_target_slices,
        )
        
        # Pad feature dimension to max_feature_dim to ensure consistent batch size
        feats1, orig_embed_dim1 = self._pad_feature_dim(feats1)
        feats2, orig_embed_dim2 = self._pad_feature_dim(feats2)
        
        assert feats1.shape[-1] == feats2.shape[-1], (
            f"Expected padded feature dim to be equal, but got {feats1.shape[-1]} and {feats2.shape[-1]}!"
        )
        
        result = {
            "feats1": feats1,                # [seq_len, max_feature_dim] or [seq_len, num_tiled_regions, max_feature_dim]
            "feats2": feats2,                # [seq_len, max_feature_dim] or [seq_len, num_tiled_regions, max_feature_dim]
            "orig_embed_dim1": torch.as_tensor(orig_embed_dim1, dtype=torch.long),
            "orig_embed_dim2": torch.as_tensor(orig_embed_dim2, dtype=torch.long),
            "seq_len": torch.as_tensor(seq_len, dtype=torch.long),
            "physical_positions": physical_positions,  # [num_target_slices] in [0, 1]
            "study_id": study_id,
            "dataset_name": study_id.split("_", 1)[0],
            "plane": selected_view_plane,
            "uid": selected_uid,
        }
        
        # Add labels for semi-supervised contrastive learning
        if self._num_labels > 0:
            dataset_name, exam_id = study_id.split("_", 1)
            label_tensor = self._annotations.get(dataset_name, {}).get(exam_id, None)
            
            if label_tensor is not None:
                result["label"] = label_tensor.clone()
                result["has_label"] = torch.tensor(True)
            else:
                # Unlabeled sample: dummy label with sentinel, has_label=False
                result["label"] = torch.zeros(self._num_labels, dtype=torch.float32)
                result["has_label"] = torch.tensor(False)
        
        return result

class FeatClassificationDataset(Dataset):
    """
    Dataset for downstream classification tasks using precomputed slice features.
    This dataset loads precomputed features and their corresponding labels for 
    linear probing evaluation of the SSL model.
    
    When multiple slice_encoder_models are provided, returns features from all encoders.
    
    Directory structure: {feat_dir}/{model_name}/{split}/{plane}/*.safetensors
    """
    def __init__(self,
                 feat_dir: str,
                 slice_encoder_models: List[str],
                 view_plane: str,
                 split: str,
                 annotations_path: str,
                 task: str,
                 target_columns: List[str],
                 cache_in_memory: bool = False):
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
            cache_in_memory: If True, preload all feature tensors into RAM at init.
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
        self.df_labels = pd.read_csv(annotations_path, dtype={"ID": str})
        self.df_labels.set_index("ID", inplace=True)
        
        # Validate target columns exist in the DataFrame
        missing_cols = [col for col in target_columns if col not in self.df_labels.columns]
        if missing_cols:
            available_cols = list(self.df_labels.columns)
            raise ValueError(
                f"Target columns {missing_cols} not found in annotations file. "
                f"Available columns: {available_cols}"
            )
        
        self.sample_ids = list(self.df_labels.index)
        
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
                    fid = fname.split(".")[0]
                    self.id_filename_map[fid] = fname

        self._feat_cache = None
        if cache_in_memory:
            unique_paths = set()
            for sample_id in self.sample_ids:
                filename = self.id_filename_map[sample_id]
                for model_name in self.slice_encoder_models:
                    unique_paths.add(os.path.join(self.feat_paths[model_name], filename))
            self._feat_cache = FeatureCache(unique_paths)

    def _get_feat(self, feat_path: str) -> Tuple[torch.Tensor, dict]:
        """Return feature tensor and metadata from cache if available, otherwise load from disk."""
        if self._feat_cache is not None:
            feat, metadata = self._feat_cache.get(feat_path)
            _validate_feature_metadata(
                feat,
                metadata,
            )
            return feat, metadata
        return self._load_feat(feat_path)

    def __len__(self):
        return len(self.sample_ids)
    
    def _load_feat(self, feat_path: str) -> Tuple[torch.Tensor, dict]:
        return _read_and_validate_feat(feat_path)
    
    def _get_label(self, sample_id: str) -> torch.Tensor:
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
        metadata = None
        
        for encoder in self.slice_encoder_models:
            feat_file = os.path.join(self.feat_paths[encoder], filename)
            if metadata is None:
                feat, metadata = self._get_feat(feat_file)
            else:
                feat, _ = self._get_feat(feat_file)
            seq_lengths.append(feat.shape[0])
            feat_embeds.append(feat)
        
        # Assert all encoders have same sequence length in the same exam (same 3D volume)
        if len(seq_lengths) > 1:
            assert all(seq_len == seq_lengths[0] for seq_len in seq_lengths), (
                f"Sequence length mismatch across slice encoders for sample {sample_id}: "
                f"{dict(zip(self.slice_encoder_models, seq_lengths))}. "
                f"All encoders should have the same number of slices for the same exam."
            )
            ranks = [feat.ndim for feat in feat_embeds]
            assert all(rank == ranks[0] for rank in ranks), (
                f"Feature dimensions mismatch across slice encoders for sample {sample_id}: "
                f"{dict(zip(self.slice_encoder_models, ranks))}. "
                "Do not mix global-only and tiled multi-crop CLS caches."
            )
            if ranks[0] == 3:
                region_counts = [feat.shape[1] for feat in feat_embeds]
                assert all(count == region_counts[0] for count in region_counts), (
                    f"Tiled region-count mismatch across slice encoders for sample {sample_id}: "
                    f"{dict(zip(self.slice_encoder_models, region_counts))}."
                )
        
        physical_positions = PrecomputedFeatPairDataset._compute_physical_positions(
            np.arange(seq_lengths[0]), seq_lengths[0], seq_lengths[0]
        )

        return {
            "feature_embeds": feat_embeds,
            "seq_length": seq_lengths[0],
            "physical_positions": physical_positions,
            "label": label,
            "sample_id": sample_id
        }


class UnlabeledFeatDataset(Dataset):
    """
    Dataset for loading precomputed features without annotations for plotting embeddings.

    Scans ``{feat_dir}/{model_name}/{split}/{plane}/*.safetensors`` and returns
    raw feature tensors with a dummy label.  Metadata (dataset name, plane) is
    tracked externally by the caller.

    Directory structure: {feat_dir}/{model_name}/{split}/{plane}/*.safetensors
    """

    def __init__(
        self,
        feat_dir: str,
        slice_encoder_models: List[str],
        split: str,
        view_plane: str,
    ):
        self.feat_dir = feat_dir
        self.slice_encoder_models = slice_encoder_models
        self.view_plane = view_plane
        self.split = split

        first_model = slice_encoder_models[0]
        feat_path = os.path.join(feat_dir, first_model, split, view_plane)
        if not os.path.exists(feat_path):
            raise FileNotFoundError(f"Feature path {feat_path} does not exist!")

        self.feat_files = sorted(glob(os.path.join(feat_path, "*.safetensors")))
        if len(self.feat_files) == 0:
            raise FileNotFoundError(f"No .safetensors files found in {feat_path}")

        self.id_filename_map = {}
        for f in self.feat_files:
            fname = os.path.basename(f)
            fid = fname.split(".")[0]
            self.id_filename_map[fid] = fname

        self.sample_ids = list(self.id_filename_map.keys())

        self.feat_paths = {}
        for model_name in slice_encoder_models:
            p = os.path.join(feat_dir, model_name, split, view_plane)
            if not os.path.exists(p):
                raise FileNotFoundError(f"Feature path {p} does not exist!")
            self.feat_paths[model_name] = p

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, idx: int) -> Dict:
        sample_id = self.sample_ids[idx]
        filename = self.id_filename_map[sample_id]

        feat_embeds = []
        seq_lengths = []
        metadata = None
        for model_name in self.slice_encoder_models:
            feat_file = os.path.join(self.feat_paths[model_name], filename)
            with safe_open(feat_file, framework="pt", device="cpu") as f:
                feat = f.get_tensor("feats")
                if metadata is None:
                    metadata = f.metadata()
            seq_lengths.append(feat.shape[0])
            feat_embeds.append(feat)

        if len(seq_lengths) > 1:
            assert all(s == seq_lengths[0] for s in seq_lengths), (
                f"Sequence length mismatch for {sample_id}: "
                f"{dict(zip(self.slice_encoder_models, seq_lengths))}"
            )

        physical_positions = PrecomputedFeatPairDataset._compute_physical_positions(
            np.arange(seq_lengths[0]), seq_lengths[0], seq_lengths[0]
        )

        return {
            "feature_embeds": feat_embeds,
            "seq_length": seq_lengths[0],
            "physical_positions": physical_positions,
            "label": torch.tensor(0, dtype=torch.long),
            "sample_id": sample_id,
        }


class MultiViewFeatClassificationDataset(Dataset):
    """
    Dataset for multi-view classification using precomputed slice features.
    Returns features from all view planes for each sample to enable end-to-end
    multi-view training.
    
    Directory structure: {feat_dir}/{model_name}/{split}/{plane}/*.safetensors
    """
    def __init__(self,
                 feat_dir: str,
                 slice_encoder_models: List[str],
                 view_planes: List[str],
                 split: str,
                 annotations_path: str,
                 task: str,
                 target_columns: List[str],
                 cache_in_memory: bool = False):
        """
        Args:
            feat_dir: Directory containing precomputed features
            slice_encoder_models: List of slice encoder model names
            view_planes: List of view planes (e.g., ["axial", "sagittal", "coronal"])
            split: Data split ("train" or "test")
            annotations_path: Path to the annotation CSV file
            task: Classification task type ("binary", "multiclass", or "multilabel")
            target_columns: List of column names to use as classification targets
            cache_in_memory: If True, preload all feature tensors into RAM at init.
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
        self.df_labels = pd.read_csv(annotations_path, dtype={"ID": str})
        self.df_labels.set_index("ID", inplace=True)
        
        # Validate target columns exist in the DataFrame
        missing_cols = [col for col in target_columns if col not in self.df_labels.columns]
        if missing_cols:
            available_cols = list(self.df_labels.columns)
            raise ValueError(
                f"Target columns {missing_cols} not found in annotations file. "
                f"Available columns: {available_cols}"
            )
        
        self.sample_ids = list(self.df_labels.index)
        
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
                        fid = fname.split(".")[0]
                        self.id_filename_map[fid] = fname

        self._feat_cache = None
        if cache_in_memory:
            unique_paths = set()
            for sample_id in self.sample_ids:
                filename = self.id_filename_map[sample_id]
                for view_plane in self.view_planes:
                    for model_name in self.slice_encoder_models:
                        unique_paths.add(os.path.join(self.feat_paths[view_plane][model_name], filename))
            self._feat_cache = FeatureCache(unique_paths)

    def _get_feat(self, feat_path: str) -> Tuple[torch.Tensor, dict]:
        """Return feature tensor and metadata from cache if available, otherwise load from disk."""
        if self._feat_cache is not None:
            feat, metadata = self._feat_cache.get(feat_path)
            _validate_feature_metadata(
                feat,
                metadata,
            )
            return feat, metadata
        return self._load_feat(feat_path)

    def __len__(self):
        return len(self.sample_ids)
    
    def _load_feat(self, feat_path: str) -> Tuple[torch.Tensor, dict]:
        return _read_and_validate_feat(feat_path)
    
    def _get_label(self, sample_id: str) -> torch.Tensor:
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
        physical_positions_by_view = {}
        
        for view_plane in self.view_planes:
            feat_embeds = []
            seq_lengths = []
            metadata = None
            
            for encoder in self.slice_encoder_models:
                feat_file = os.path.join(self.feat_paths[view_plane][encoder], filename)
                if metadata is None:
                    feat, metadata = self._get_feat(feat_file)
                else:
                    feat, _ = self._get_feat(feat_file)
                seq_lengths.append(feat.shape[0])
                feat_embeds.append(feat)
            
            # Assert all encoders have same sequence length for this view
            if len(seq_lengths) > 1:
                assert all(seq_len == seq_lengths[0] for seq_len in seq_lengths), (
                    f"Sequence length mismatch across slice encoders for sample {sample_id}, view {view_plane}"
                )
                ranks = [feat.ndim for feat in feat_embeds]
                assert all(rank == ranks[0] for rank in ranks), (
                    f"Feature dimensions mismatch across slice encoders for sample {sample_id}, "
                    f"view {view_plane}: {dict(zip(self.slice_encoder_models, ranks))}. "
                    "Do not mix global-only and tiled multi-crop CLS caches."
                )
                if ranks[0] == 3:
                    region_counts = [feat.shape[1] for feat in feat_embeds]
                    assert all(count == region_counts[0] for count in region_counts), (
                        f"Tiled region-count mismatch across slice encoders for sample {sample_id}, "
                        f"view {view_plane}: {dict(zip(self.slice_encoder_models, region_counts))}."
                    )
            
            features_by_view[view_plane] = feat_embeds
            seq_lengths_by_view[view_plane] = seq_lengths[0]
            physical_positions_by_view[view_plane] = PrecomputedFeatPairDataset._compute_physical_positions(
                np.arange(seq_lengths[0]), seq_lengths[0], seq_lengths[0]
            )
        
        return {
            "feature_embeds": features_by_view,  
            "seq_length": seq_lengths_by_view,  
            "physical_positions": physical_positions_by_view,
            "label": label,
            "sample_id": sample_id,
        }
