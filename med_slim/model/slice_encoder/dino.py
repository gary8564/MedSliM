import torch
import torch.nn as nn 
import logging
from transformers import AutoModel
from einops import rearrange
from pathlib import Path
from med_slim.logging.setup import init_logging
init_logging()
logger = logging.getLogger(__name__)

class DinoFeatureExtractor(nn.Module):
    def __init__(self, 
                 model_repo: str,
                 local_cache_dir: str = None):
        """
        Initialize the DINO classifier.
        
        Args:
            model_repo: Pretrained model repository on Hugging Face (e.g., "facebook/dinov2-base", "facebook/dinov3-vit7b16-pretrain-lvd1689m", "microsoft/rad-dino")
            local_cache_dir: Local directory to cache the model. If None, the model will be cached in the default Hugging Face cache directory.
        """
        super().__init__()
        if local_cache_dir is not None:
            local_cache_dir = Path(local_cache_dir)
            local_cache_dir.mkdir(parents=True, exist_ok=True)
        self.model = AutoModel.from_pretrained(model_repo, cache_dir=local_cache_dir)
    
    def forward(self, x):
        """
        Forward pass through the pretrained DINO model.
        
        Args:
            x: Input tensor, has shape [B, C, W, H, D]
            
        Returns:
            features: Features representation from the DINO model
        """
        x = x.swapaxes(2, -1) # [B, C, W, H, D] -> [B, C, D, H, W]
        B, C, *_ = x.shape
        assert C == 1, "MRI/CT slices should be grayscale (C = 1)"
        x = rearrange(x, 'b c d h w -> (b d c) h w') 
        x = x[:, None, :, :] # [B*D, H, W] -> [B*D, 1, H, W]
        x = x.repeat(1, 3, 1, 1) # Gray to RGB
        outputs = self.model(x, return_dict=True)
        
        # Extract cls token from the last hidden state as the feature representation
        features = outputs.pooler_output # [(B D), C, H, W] -> [(B D), embed_dim] 
        features = rearrange(features, '(b d) e -> b d e', b=B) # [(B D), embed_dim] -> [B, D, embed_dim]
        return features
    
if __name__ == "__main__":
    
    model_repo = "facebook/dinov3-convnext-large-pretrain-lvd1689m" #"facebook/dinov2-base"
    # Alternative: model_repo = "microsoft/rad-dino"
    local_cache_dir = "/work/qj474765/models"
    # Initialize the feature extractor with the pretrained model
    feature_extractor = DinoFeatureExtractor(model_repo, local_cache_dir)
    
    # Freeze the model
    for param in feature_extractor.model.parameters():
        param.requires_grad = False
    
    # Verify the number of transformer layers
    num_layers = feature_extractor.model.config.num_hidden_layers
    print(f"Number of transformer blocks: {num_layers}")
    
    # Print model statistics
    total_params = sum(p.numel() for p in feature_extractor.model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(p.numel() for p in feature_extractor.model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")