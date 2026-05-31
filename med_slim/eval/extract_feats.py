"""
Feature extraction of MRI/CT volumetric images using COBRA.

Adapted from COBRA (https://github.com/KatherLab/COBRA/blob/main/cobra/inference/extract_feats.py)
==================================================================================================
COBRA (Histopathology):                         MedSliM (MRI/CT):
  tiles                                         slices
    ↓ ABMIL                                        ↓ ABMIL / cross-attention
  slide embedding                               volume embedding (per view plane / MRI sequence)
    ↓ concat + ABMIL                               ↓ logistic regression classifier
  patient embedding                             patient embedding (multi-view plane / MRI sequence)
==================================================================================================

References:
Lenz, Tim, Peter Neidlinger, Marta Ligero, Georg Wölflein, Marko van Treeck and Jakob Nikolas Kather. 
Unsupervised Foundation Model-Agnostic Slide-Level Representation Learning.
2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR): 30807-30817, 2024.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from typing import Tuple, List, Optional
from tqdm import tqdm

from med_slim.model.sequence_encoder.cobra import Cobra


def get_volume_feats(
    cobra_model: Cobra,
    dataloader: DataLoader,
    accelerator: Accelerator,
) -> Tuple[torch.Tensor, torch.Tensor, List]:
    """
    Extract volume-level embeddings using pretrained COBRA model.
    
    Args:
        cobra_model: Pretrained COBRA model in inference mode
        dataloader: DataLoader yielding batches of slice features from all encoders
        accelerator: HuggingFace Accelerator
    
    Returns:
        volume_feats: [N, embed_dim] tensor of volume-level features
        labels: [N] tensor of labels
        sample_ids: List of sample IDs
    """
    cobra_model.eval()
    all_volume_feats = []
    all_labels = []
    all_sample_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting volume embeddings", disable=not accelerator.is_main_process):
            labels = batch["labels"]  
            sample_ids = batch["sample_ids"]
            seq_lengths = batch["seq_lengths"].to(accelerator.device)
            physical_positions = batch.get("physical_positions")
            if physical_positions is not None:
                physical_positions = physical_positions.to(accelerator.device, dtype=torch.float32)
            
            # Get features from all encoders and cast to model dtype
            encoder_feats = [f.to(accelerator.device, dtype=next(cobra_model.parameters()).dtype) for f in batch["features"]]
            
            # COBRA embeds each encoder's features and averages them across encoders
            volume_feats = cobra_model(
                encoder_feats,
                seq_lengths=seq_lengths,
                physical_positions=physical_positions,
            )
            
            # Cast to float32 for downstream eval 
            all_volume_feats.append(volume_feats.float())
            all_labels.append(labels)
            all_sample_ids.extend(sample_ids)
    
    volume_feats = torch.cat(all_volume_feats, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    return volume_feats, labels, all_sample_ids


def get_patient_feats(
    cobra_model: Cobra,
    multiview_dataloader: DataLoader,
    accelerator: Accelerator,
    aggregation: str = "mean",
) -> Tuple[torch.Tensor, torch.Tensor, List]:
    """
    Extract patient-level embeddings by aggregating multiple volumes.
    
    When a patient has multiple view planes (axial, sagittal, coronal) 
    or multiple MRI sequences (T1, T2, FLAIR), this aggregates them into 
    a single patient-level embedding.
    
    Args:
        cobra_model: Pretrained COBRA model in inference mode
        multiview_dataloader: DataLoader yielding batches from MultiViewFeatClassificationDataset
        accelerator: HuggingFace Accelerator
        aggregation: aggregation method for view embeddings ("mean" or "max")
    
    Returns:
        patient_feats: [N, embed_dim] tensor of patient-level features
        labels: [N] tensor of labels
        sample_ids: List of sample IDs
    """
    cobra_model.eval()
    all_patient_feats = []
    all_labels = []
    all_sample_ids = []
    
    with torch.no_grad():
        for batch in tqdm(multiview_dataloader, desc="Extracting patient embeddings", disable=not accelerator.is_main_process):
            labels = batch["labels"]
            sample_ids = batch["sample_ids"]
            features_by_view = batch["features"]
            seq_lengths_by_view = batch["seq_lengths"]
            physical_positions_by_view = batch.get("physical_positions", {})
            
            view_names = list(features_by_view.keys())
            model_dtype = next(cobra_model.parameters()).dtype
            
            view_embeddings = []
            for view_plane in view_names:
                encoder_feats = [f.to(accelerator.device, dtype=model_dtype) 
                                for f in features_by_view[view_plane]]
                seq_lengths = seq_lengths_by_view[view_plane].to(accelerator.device)
                physical_positions = physical_positions_by_view.get(view_plane)
                if physical_positions is not None:
                    physical_positions = physical_positions.to(accelerator.device, dtype=torch.float32)
                view_emb = cobra_model(
                    encoder_feats,
                    seq_lengths=seq_lengths,
                    physical_positions=physical_positions,
                )
                view_embeddings.append(view_emb)
            
            stacked_views = torch.stack(view_embeddings, dim=1)
            
            if aggregation == "mean":
                patient_emb = stacked_views.mean(dim=1)
            elif aggregation == "max":
                patient_emb = stacked_views.max(dim=1).values
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")
            
            all_patient_feats.append(patient_emb.float())
            all_labels.append(labels)
            all_sample_ids.extend(sample_ids)
    
    patient_feats = torch.cat(all_patient_feats, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    return patient_feats, labels, all_sample_ids


def get_volume_attention(
    cobra_model: Cobra,
    dataloader: DataLoader,
    accelerator: Accelerator,
    max_samples: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str], List[int]]:
    """
    Extract slice-level attention weights using pretrained COBRA model.
    
    Supports both ABMIL and cross-attention slice pooling modes.

    Args:
        cobra_model: Pretrained COBRA model in inference mode
        dataloader: DataLoader yielding batches of slice features
        accelerator: HuggingFace Accelerator
        max_samples: Maximum number of samples to process (None = all)
    
    Returns:
        attention_weights: List of attention arrays [num_slices] per sample
        labels: List of label arrays per sample
        sample_ids: List of sample IDs
        seq_lengths: List of sequence lengths
    """
    cobra_model.eval()
    all_attention = []
    all_labels = []
    all_sample_ids = []
    all_seq_lengths = []
    
    sample_count = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting attention", disable=not accelerator.is_main_process):
            seq_lengths = batch["seq_lengths"].to(accelerator.device)
            physical_positions = batch.get("physical_positions")
            if physical_positions is not None:
                physical_positions = physical_positions.to(accelerator.device, dtype=torch.float32)
            features = [f.to(accelerator.device, dtype=next(cobra_model.parameters()).dtype) 
                       for f in batch["features"]]
            
            # Get per-head attention and aggregate with min across heads.
            # Min-attention is standard in medical imaging explainability
            # as it highlights slices that ALL heads agree are important.
            attention = cobra_model(
                features,
                seq_lengths=seq_lengths,
                physical_positions=physical_positions,
                get_per_head_attention=True,
            )
            # attention shape: [B, num_heads, max_seq_len]
            attention = attention.min(dim=1).values  # [B, max_seq_len]
            attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            attention = attention.cpu().numpy()
            
            batch_size = attention.shape[0]
            for i in range(batch_size):
                if max_samples and sample_count >= max_samples:
                    break
                
                seq_len = seq_lengths[i].item()
                attn = attention[i, :seq_len]  # Trim to actual sequence length
                
                all_attention.append(attn)
                all_labels.append(batch["labels"][i].cpu().numpy())
                all_sample_ids.append(batch["sample_ids"][i])
                all_seq_lengths.append(seq_len)
                sample_count += 1
            
            if max_samples and sample_count >= max_samples:
                break
    
    return all_attention, all_labels, all_sample_ids, all_seq_lengths


def get_volume_region_attention(
    cobra_model: Cobra,
    dataloader: DataLoader,
    accelerator: Accelerator,
    max_samples: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str], List[int]]:
    """
    Extract within-slice region attention weights from a tiled multi-crop CLS COBRA model.

    For each slice, the global CLS token attends over the regional crop tokens; the
    resulting distribution shows which quadrant the model focused on per slice. This is
    complementary to the slice-level attention from ``get_volume_attention`` and should
    always be interpreted alongside it (a region can dominate a slice the volume ignores).

    Args:
        cobra_model: Pretrained COBRA model built with regional_tokens > 0.
        dataloader: DataLoader yielding batches of tiled slice features
            [B, num_slices, num_tiled_regions, embed_dim].
        accelerator: HuggingFace Accelerator.
        max_samples: Maximum number of samples to process (None = all).

    Returns:
        region_attention: List of arrays [num_slices, num_tiled_regions-1] per sample.
        labels: List of label arrays per sample.
        sample_ids: List of sample IDs.
        seq_lengths: List of sequence lengths.
    """
    cobra_model.eval()
    all_attention = []
    all_labels = []
    all_sample_ids = []
    all_seq_lengths = []
    sample_count = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting region attention", disable=not accelerator.is_main_process):
            seq_lengths = batch["seq_lengths"].to(accelerator.device)
            physical_positions = batch.get("physical_positions")
            if physical_positions is not None:
                physical_positions = physical_positions.to(accelerator.device, dtype=torch.float32)
            features = [f.to(accelerator.device, dtype=next(cobra_model.parameters()).dtype)
                        for f in batch["features"]]

            region_attn = cobra_model(
                features,
                seq_lengths=seq_lengths,
                physical_positions=physical_positions,
                get_region_attention=True,
            )  # [B, num_slices, 1, num_tiled_regions-1]
            region_attn = region_attn.squeeze(2).cpu().numpy()  # [B, num_slices, num_tiled_regions-1]

            batch_size = region_attn.shape[0]
            for i in range(batch_size):
                if max_samples and sample_count >= max_samples:
                    break
                seq_len = seq_lengths[i].item()
                all_attention.append(region_attn[i, :seq_len])  # [seq_len, num_tiled_regions-1]
                all_labels.append(batch["labels"][i].cpu().numpy())
                all_sample_ids.append(batch["sample_ids"][i])
                all_seq_lengths.append(seq_len)
                sample_count += 1

            if max_samples and sample_count >= max_samples:
                break

    return all_attention, all_labels, all_sample_ids, all_seq_lengths


def get_volume_attention_per_head(
    cobra_model: Cobra,
    dataloader: DataLoader,
    accelerator: Accelerator,
    max_samples: Optional[int] = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str], List[int], int]:
    """
    Extract per-head slice-level attention weights from COBRA ABMIL pooling.
    
    Args:
        cobra_model: Pretrained COBRA model with ABMIL slice pooling
        dataloader: DataLoader yielding batches of slice features
        accelerator: HuggingFace Accelerator
        max_samples: Maximum number of samples to process (None = all)
    
    Returns:
        attention_weights: List of attention arrays [num_heads, num_slices] per sample
        labels: List of label arrays per sample
        sample_ids: List of sample IDs
        seq_lengths: List of sequence lengths
        num_heads: Number of attention heads
    """
    cobra_model.eval()
    all_attention = []
    all_labels = []
    all_sample_ids = []
    all_seq_lengths = []
    num_heads = 0

    sample_count = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting per-head attention", disable=not accelerator.is_main_process):
            seq_lengths = batch["seq_lengths"].to(accelerator.device)
            physical_positions = batch.get("physical_positions")
            if physical_positions is not None:
                physical_positions = physical_positions.to(accelerator.device, dtype=torch.float32)
            features = [f.to(accelerator.device, dtype=next(cobra_model.parameters()).dtype)
                       for f in batch["features"]]

            attention = cobra_model(
                features,
                seq_lengths=seq_lengths,
                physical_positions=physical_positions,
                get_per_head_attention=True,
            )
            # attention shape: [B, num_heads, max_seq_len]
            attention = attention.cpu().numpy()
            num_heads = attention.shape[1]

            batch_size = attention.shape[0]
            for i in range(batch_size):
                if max_samples and sample_count >= max_samples:
                    break

                seq_len = seq_lengths[i].item()
                attn = attention[i, :, :seq_len]  # [num_heads, seq_len]

                all_attention.append(attn)
                all_labels.append(batch["labels"][i].cpu().numpy())
                all_sample_ids.append(batch["sample_ids"][i])
                all_seq_lengths.append(seq_len)
                sample_count += 1

            if max_samples and sample_count >= max_samples:
                break

    return all_attention, all_labels, all_sample_ids, all_seq_lengths, num_heads
