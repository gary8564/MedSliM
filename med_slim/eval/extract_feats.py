"""
Adapted from https://github.com/KatherLab/COBRA/blob/main/cobra/inference/extract_feats.py

Lenz, Tim, Peter Neidlinger, Marta Ligero, Georg Wölflein, Marko van Treeck and Jakob Nikolas Kather. 
Unsupervised Foundation Model-Agnostic Slide-Level Representation Learning.
2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR): 30807-30817, 2024.
"""

import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from typing import Tuple, List
from tqdm import tqdm
from med_slim.model.sequence_encoder.cobra import Cobra


def get_cobra_feats(
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
        cobra_feats: [N, embed_dim] tensor of volume-level COBRA features
        labels: [N] tensor of labels
        sample_ids: List of sample IDs
    """
    cobra_model.eval()
    all_cobra_feats = []
    all_labels = []
    all_sample_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting COBRA embeddings", disable=not accelerator.is_main_process):
            labels = batch["labels"]  
            sample_ids = batch["sample_ids"]
            
            # Get features from all encoders and cast to model dtype
            encoder_feats = [f.to(accelerator.device, dtype=next(cobra_model.parameters()).dtype) for f in batch["feature_embeds"]]
            
            # COBRA embeds each encoder's features and average them across encoders
            cobra_feats = cobra_model(encoder_feats) 
            
            # Cast to float32 for downstream eval 
            all_cobra_feats.append(cobra_feats.float())
            all_labels.append(labels)
            all_sample_ids.extend(sample_ids)
    
    cobra_feats = torch.cat(all_cobra_feats, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    return cobra_feats, labels, all_sample_ids

