import torch.nn as nn
import torch
import torch.nn.functional as F
import logging
from typing import Callable, Optional
from open_clip import create_model_from_pretrained
from einops import rearrange
from pathlib import Path

from med_slim.logging.setup import init_logging
init_logging()
logger = logging.getLogger(__name__)

class CLIPFeatureExtractor(nn.Module):
    def __init__(self, 
                 model_repo: str,
                 local_cache_dir: str = None):
        """
        Initialize the BiomedCLIP model.

        Args:
            model_repo: Pre-trained BiomedCLIP model repository on Hugging Face (e.g., "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")
            local_cache_dir: Local cache directory to store the model. If None, the model will be cached in the default Hugging Face cache directory.
        """
        super().__init__()
        if local_cache_dir is not None:
            local_cache_dir = Path(local_cache_dir)
            local_cache_dir.mkdir(parents=True, exist_ok=True)
        # Load BiomedCLIP model from Hugging Face hub using open_clip
        self.model, _ = create_model_from_pretrained(f'hf-hub:{model_repo}', cache_dir=local_cache_dir)
        
    def forward(self, pixel_values):
        """
        Forward pass through the BiomedCLIP model.
        
        Args:
            pixel_values: Input tensor, has shape [B, C, W, H, D]
            
        Returns:
            features: Normalized image features from the BiomedCLIP vision encoder
        """
        pixel_values = pixel_values.swapaxes(2, -1) # [B, C, W, H, D] -> [B, C, D, H, W]
        B, C, *_ = pixel_values.shape
        assert C == 1, "MRI/CT slices should be grayscale (C = 1)"
        pixel_values = rearrange(pixel_values, 'b c d h w -> (b d c) h w') 
        pixel_values = pixel_values[:, None, :, :] # [B*D, H, W] -> [B*D, 1, H, W]
        pixel_values = pixel_values.repeat(1, 3, 1, 1) # Gray to RGB
        # Encode images and normalize features
        image_features = self.model.encode_image(pixel_values) # [B*D, embed_dim]
        features = image_features / image_features.norm(dim=-1, keepdim=True) # [B*D, embed_dim] -> [B*D, embed_dim]
        features = rearrange(features, '(b d) e -> b d e', b=B) # [B*D, embed_dim] -> [B, D, embed_dim]
        return features

if __name__ == "__main__":
    import os
        
    # Load pre-trained BiomedCLIP model
    feature_extractor = CLIPFeatureExtractor(model_repo='microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    
    # Freeze the model
    for name, param in feature_extractor.model.named_parameters():
        param.requires_grad = False
    
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in feature_extractor.model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in feature_extractor.model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
