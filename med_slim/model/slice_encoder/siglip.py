import torch.nn as nn
import torch
import logging
from transformers import AutoModel
from einops import rearrange
from pathlib import Path
from med_slim.logging.setup import init_logging
init_logging()
logger = logging.getLogger(__name__)

class SigLipFeatureExtractor(nn.Module):
    def __init__(self, 
                 model_repo: str,
                 local_cache_dir: str = None):
        """
        Initialize the MedSigLIP    .
        
        Args:
            model_repo: Pre-trained MedSigLIP model repository on Hugging Face (e.g., "google/medsiglip-448")
            local_cache_dir: Local cache directory to store the model. If None, the model will be cached in the default Hugging Face cache directory.
        """
        super().__init__()
        if local_cache_dir is not None:
            local_cache_dir = Path(local_cache_dir)
            local_cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = AutoModel.from_pretrained(model_repo, cache_dir=local_cache_dir)
      
    def forward(self, pixel_values):
        """
        Forward pass through the MedSigLIP model.
        
        Args:
            pixel_values: Input tensor, has shape [B, C, W, H, D]
            
        Returns:
            features: Features representation from the MedSigLIP model
        """
        pixel_values = pixel_values.swapaxes(2, -1) # [B, C, W, H, D] -> [B, C, D, H, W]
        B, C, *_ = pixel_values.shape
        assert C == 1, "MRI/CT slices should be grayscale (C = 1)"
        pixel_values = rearrange(pixel_values, 'b c d h w -> (b d c) h w') 
        pixel_values = pixel_values[:, None, :, :] # [B*D, H, W] -> [B*D, 1, H, W]
        pixel_values = pixel_values.repeat(1, 3, 1, 1) # Gray to RGB
        # Extract features and normalize
        vision_outputs = self.model.vision_model(
            pixel_values=pixel_values,
            return_dict=True
        )    
        features = vision_outputs.pooler_output / vision_outputs.pooler_output.norm(dim=-1, keepdim=True)
        features = rearrange(features, '(b d) e -> b d e', b=B) # [(B D), embed_dim] -> [B, D, embed_dim]
        return features

if __name__ == "__main__":
    # Load pre-trained MedSigLIP model
    feature_extractor = SigLipFeatureExtractor(model_repo='google/medsiglip-448')
    
    # Freeze the model
    for name, param in feature_extractor.model.named_parameters():
        param.requires_grad = False
    
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in feature_extractor.model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in feature_extractor.model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    
