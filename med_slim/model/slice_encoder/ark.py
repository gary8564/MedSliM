import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import timm.models.swin_transformer as swin
import timm
from typing import Optional
from einops import rearrange

from med_slim.logging.setup import init_logging
init_logging()
logger = logging.getLogger(__name__)

TIMM_VERSION = timm.__version__

class SwinTransformer(swin.SwinTransformer):
    def __init__(self, 
                 num_classes_list: list[int],
                 img_size: int = 768,
                 patch_size: int = 4,
                 window_size: int = 12,
                 embed_dim: int = 192,
                 depths: tuple = (2, 2, 18, 2),
                 num_heads: tuple = (6, 12, 24, 48),
                 projector_features: int = 1376,
                 use_mlp: bool = False):
        """
        Initialize Swin Transformer for Ark with attention map support.
        
        Args:
            num_classes_list: List of number of output classes for each pretrained classification task
            img_size: Input image size
            patch_size: Patch size for embedding
            window_size: Window size for Swin Transformer
            embed_dim: Embedding dimension
            depths: Number of layers in each stage
            num_heads: Number of attention heads in each stage
            projector_features: Dimension for projector
            use_mlp: Whether to use MLP projector
        """
        super().__init__(
            num_classes=0,  # Handle classification separately
            img_size=img_size,
            patch_size=patch_size,
            window_size=window_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads
        )
        assert num_classes_list is not None
        self.num_classes_list = num_classes_list
        
        # Initialize projector
        self.encoder_features = self.num_features
        self.num_features = projector_features
        if use_mlp:
            self.projector = nn.Sequential(
                nn.Linear(self.encoder_features, self.num_features),
                nn.ReLU(inplace=True),
                nn.Linear(self.num_features, self.num_features)
            )
        else:
            self.projector = nn.Linear(self.encoder_features, self.num_features)
        
        # Initialize omini classification head
        self.omni_heads = []
        for num_classes in self.num_classes_list:
            self.omni_heads.append(nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity())
        self.omni_heads = nn.ModuleList(self.omni_heads)
        # Freeze omni_heads so they don't participate in grads during downstream training
        for p in self.omni_heads.parameters():
            p.requires_grad = False
    
    def forward_features(self, x):
        """Extract features from the pretrained Ark model with optional attention maps."""
        x = super().forward_features(x)
        
        # Handle compatibility between timm v0.5.4 and latest version
        # timm 0.5.x -> (B, C)
        # timm >= 0.8.x -> (B, L, C) or (B, H, W, C)
        if x.ndim == 3:           # (B, L, C)
            # logger.info(f"timm version {TIMM_VERSION}: (B, L, C) -> (B, C) need to be handled manually!")
            x = x.transpose(1, 2)         # (B, C, L)
            x = F.adaptive_avg_pool1d(x, 1)  # (B, C, 1)
            x = x.flatten(1)              # (B, C)
        elif x.ndim == 4:         # (B, H, W, C)
            # logger.info(f"timm version {TIMM_VERSION}: (B, H, W, C) -> (B, C) need to be handled manually!")
            x = x.permute(0, 3, 1, 2)     # (B, C, H, W)
            x = F.adaptive_avg_pool2d(x, 1)  # (B, C, 1, 1)
            x = x.flatten(1)              # (B, C)
        
        return x
    
    def forward(self, x, head_n: Optional[int] = None):
        """Forward pass through the pretrained Ark model."""
        x = self.forward_features(x)
        x = self.projector(x)
        
        if head_n is not None:
            return self.omni_heads[head_n](x), None
        else:
            return [head(x) for head in self.omni_heads]
    
    def generate_embeddings(self, x, after_proj: bool = True):
        """
        Generate embeddings for downstream tasks.
        
        Args:
            x: Input tensor
            after_proj: Whether to apply projection after feature extraction
            
        Returns:
            embeddings: Features representation from the Ark model
        """
        x = self.forward_features(x)
        if after_proj:
            x = self.projector(x)
        
        return x
    
    def get_feature_dimension(self, after_proj: bool = True) -> int:
        """
        Get the feature dimension for the current configuration.
        
        Args:
            after_proj: Whether to return dimension after projection
            
        Returns:
            Feature dimension
        """
        if after_proj:
            return self.num_features
        else:
            return self.encoder_features


class ArkFeatureExtractor(nn.Module):
    def __init__(self, 
                 model_checkpoint_path: str,
                 device: str = "cpu",
                 use_projector: bool = False):
        """
        Initialize the Ark feature extractor.
        
        Args:
            model_checkpoint_path: Path to the pretrained Ark checkpoint file
            device: "cpu", "cuda", or "cuda:id"
            use_projector: Whether to use the model projector. 
                          - For linear probing, use the feature dimension from the model after projection.
                          - For fine-tuning, use the feature dimension from the model before projection.
        """
        super().__init__()
        self.model_checkpoint_path = model_checkpoint_path
        self.device = device
        self._load_pretrained_ark_model()    
        self.use_projector = use_projector
        if use_projector:
            # For linear probing, use the feature dimension from the model after projection
            self.embed_dim = self.model.num_features
        else:
            # For fine-tuning, use the feature dimension from the model before projection
            self.embed_dim = self.model.num_features if self.model.projector is None else self.model.projector.in_features  
    
    def _load_pretrained_ark_model(self,
                                   num_classes_list: list[int] = [14, 14, 14, 3, 6, 1],
                                   img_size: int = 768,
                                   patch_size: int = 4,
                                   window_size: int = 12,
                                   embed_dim: int = 192,
                                   depths: tuple = (2, 2, 18, 2),
                                   num_heads: tuple = (6, 12, 24, 48),
                                   projector_features: int = 1376,
                                   use_mlp: bool = False):
        """
        Load a pre-trained Ark model from checkpoint.
        
        Args:
            num_classes_list: List of number of output classes for each pretrained classification task
            img_size: Input image size
            patch_size: Patch size for embedding
            window_size: Window size for Swin Transformer
            embed_dim: Embedding dimension
            depths: Number of layers in each stage
            num_heads: Number of attention heads in each stage
            projector_features: Dimension for projector (if None, no projector)
            use_mlp: Whether to use MLP projector
        """
        # Initialize Ark (i.e. SwinTransformer) model
        self.model = SwinTransformer(
            num_classes_list=num_classes_list,
            img_size=img_size,
            patch_size=patch_size,
            window_size=window_size,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            projector_features=projector_features,
            use_mlp=use_mlp,
        )
        
        # Load checkpoint
        checkpoint = torch.load(self.model_checkpoint_path, map_location=self.device, weights_only=False)    
        state_dict = checkpoint["teacher"]

        # Remove "module." prefix if present (for DataParallel models)
        if any([True if 'module.' in k else False for k in state_dict.keys()]):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items() if k.startswith('module.')}
        
        # Remove unnecessary keys
        keys_to_delete = []
        for k in state_dict.keys():
            if "attn_mask" in k or k in ["head.weight", "head.bias"]:
                keys_to_delete.append(k)
        # Delete identified keys
        for k in keys_to_delete:
            if k in state_dict: # Ensure the key exists
                del state_dict[k]
        
        # Handle compatibility between timm v0.5.4 and latest version:
        # Map old layer names to new layer names
        new_state_dict = swin.checkpoint_filter_fn(state_dict, self.model)
        
        # Load state dict
        msg = self.model.load_state_dict(new_state_dict, strict=False)
        logger.info(f'Loaded Ark model with msg: {msg}')
        
    def forward(self, x):
        """
        Forward pass through the Ark feature extractor.
        
        Args:
            x: Input tensor of shape [batch_size, channels, width, height, depth]
               
        Returns:
            features: Features representation from the Ark model
        """
        x = x.swapaxes(2, -1) # [B, C, W, H, D] -> [B, C, D, H, W]
        B, C, *_ = x.shape
        assert C == 1, "MRI/CT slices should be grayscale (C = 1)"
        x = rearrange(x, 'b c d h w -> (b d c) h w') 
        x = x[:, None, :, :] # [B*D, H, W] -> [B*D, 1, H, W]
        x = x.repeat(1, 3, 1, 1) # Gray to RGB
        # Micro-batch over slices to avoid CUDA out of memory
        total_slices = x.shape[0]  # B*D
        chunk_size = 2
        outputs = []
        for start in range(0, total_slices, chunk_size):
            end = min(start + chunk_size, total_slices)
            x_chunk = x[start:end]
            out_chunk = self.model.generate_embeddings(x_chunk, after_proj=self.use_projector)
            outputs.append(out_chunk)
        features = torch.cat(outputs, dim=0)
        features = rearrange(features, '(b d) e -> b d e', b=B) # [(B D), embed_dim] -> [B, D, embed_dim]
        return features
    
if __name__ == "__main__":
    import os
    
    # Pretrained Ark checkpoint
    checkpoint_path = "/work/rwth1833/models/ark/Ark+_Nature/Ark6_swinLarge768_ep50.pth.tar"
    
    # Load the pre-trained Ark model
    feature_extractor = ArkFeatureExtractor(
        model_checkpoint_path=checkpoint_path,
        device="cpu",
        use_projector=True
    )
    
    for name, param in feature_extractor.model.named_parameters():
        param.requires_grad = False

    # Print model statistics
    total_params = sum(p.numel() for p in feature_extractor.model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in feature_extractor.model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    