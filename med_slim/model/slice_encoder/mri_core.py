"""
MRI-CORE feature extractor wrapper.

Wraps the MRI-CORE foundation model (https://github.com/mazurowski-lab/mri_foundation)
for use as a slice-level feature extractor in MedSliM.

MRI-CORE is a ViT-B/16 pretrained on 6.9M MRI slices using DINOv2, initialized
from SAM weights. The saved checkpoint is a standard DINOv2 teacher checkpoint
containing ViT backbone weights.

Reference:
    Dong et al., "MRI-CORE: A Foundation Model for Magnetic Resonance Imaging", 2024.
    arXiv: 2506.12186
"""

import torch
import torch.nn as nn
import logging
import timm
from pathlib import Path
from einops import rearrange

from med_slim.logging.setup import init_logging
init_logging()
logger = logging.getLogger(__name__)


class MriCoreFeatureExtractor(nn.Module):
    """
    Feature extractor using the MRI-CORE pretrained ViT-B/16 encoder.
    
    Loads the DINOv2 teacher checkpoint into a standard timm ViT-B/16 (224x224)
    and extracts the CLS token as the per-slice embedding (768-dim), consistent
    with how DINOv2 uses the CLS token for self-distillation loss.
    
    Input:  [B, C, W, H, D] grayscale MRI volume (C=1)
    Output: [B, D, 768] per-slice feature embeddings
    """
    
    EMBED_DIM = 768  # ViT-B hidden dimension
    
    def __init__(self, checkpoint_path: str):
        """
        Args:
            checkpoint_path: Path to the MRI-CORE checkpoint file (DINOv2 teacher checkpoint).
        """
        super().__init__()
        
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(
                f"MRI-CORE checkpoint not found at: {checkpoint_path}\n"
                "Download from: https://github.com/mazurowski-lab/mri_foundation"
            )
        
        self.embed_dim = self.EMBED_DIM
        
        # Create a standard ViT-B/16 at 224x224 (matching the DINOv2 pretraining resolution)
        self.model = timm.create_model(
            'vit_base_patch16_224',
            pretrained=False,
            num_classes=0,  # Remove classification head
        )
        
        # Load MRI-CORE weights
        self._load_checkpoint(checkpoint_path)
        
        logger.info(
            f"Loaded MRI-CORE ViT-B/16 from {checkpoint_path} "
            f"(img_size=224, embed_dim={self.EMBED_DIM})"
        )
    
    def _load_checkpoint(self, checkpoint_path: str):
        """
        Load MRI-CORE's DINOv2 teacher checkpoint into the timm ViT model.
        
        Handles key remapping:
        - Removes 'backbone.' prefix
        - Flattens chunked blocks: blocks.chunk_id.block_id.* → blocks.block_id.*
        - Skips DINOv2 projection head (dino_head.*) and mask token
        """
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        
        if 'teacher' not in ckpt:
            raise ValueError(
                f"Expected DINOv2 teacher checkpoint with 'teacher' key, "
                f"but found keys: {list(ckpt.keys())}"
            )
        
        state_dict = ckpt['teacher']
        remapped = {}
        
        for k, v in state_dict.items():
            # Skip DINOv2 projection head and mask token
            if k.startswith('dino_head.') or k == 'backbone.mask_token':
                continue
            
            # Remove 'backbone.' prefix
            new_k = k.replace('backbone.', '', 1)
            
            # Flatten chunked blocks
            parts = new_k.split('.')
            if (len(parts) > 2
                and parts[0] == 'blocks'
                and parts[1].isdigit()
                and parts[2].isdigit()):
                parts = [parts[0]] + parts[2:]  # Remove chunk index
                new_k = '.'.join(parts)
            
            remapped[new_k] = v
        
        msg = self.model.load_state_dict(remapped, strict=False)
        
        if msg.unexpected_keys:
            logger.warning(f"Unexpected keys in MRI-CORE checkpoint: {msg.unexpected_keys}")
        if msg.missing_keys:
            # Some timm-specific keys (e.g., attn_drop, proj_drop) may be missing
            non_trivial_missing = [k for k in msg.missing_keys if 'drop' not in k and 'ls' not in k]
            if non_trivial_missing:
                logger.warning(f"Missing keys when loading MRI-CORE: {non_trivial_missing}")
    
    @staticmethod
    def _minmax_normalize(x: torch.Tensor) -> torch.Tensor:
        """
        Per-slice min-max normalization to [0, 1].
        
        MRI-CORE was pretrained on min-max normalized slices, so we replicate
        this normalization at inference time. Each slice is normalized independently.
        
        Args:
            x: [N, 1, H, W] grayscale slices
            
        Returns:
            Normalized tensor in [0, 1] range
        """
        x_min = x.amin(dim=(-2, -1), keepdim=True)
        x_max = x.amax(dim=(-2, -1), keepdim=True)
        denom = (x_max - x_min).clamp_min(1e-8)
        return (x - x_min) / denom

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MRI-CORE feature extractor.
        
        Args:
            x: Input tensor of shape [B, C, W, H, D] (grayscale MRI volume, C=1)
               
        Returns:
            features: [B, D, 768] per-slice feature embeddings (CLS token)
        """
        x = x.swapaxes(2, -1)  # [B, C, W, H, D] -> [B, C, D, H, W]
        B, C, *_ = x.shape
        assert C == 1, "MRI/CT slices should be grayscale (C = 1)"
        
        # Rearrange to process slices: [B, 1, D, H, W] -> [B*D, 1, H, W]
        x = rearrange(x, 'b c d h w -> (b d) c h w')
        
        # Apply per-slice min-max normalization to [0, 1] (matching MRI-CORE pretraining)
        x = self._minmax_normalize(x)
        
        # Convert grayscale to RGB
        x = x.repeat(1, 3, 1, 1)  # [B*D, 1, H, W] -> [B*D, 3, H, W]
        
        # forward_features returns [B*D, 197, 768] (CLS + 196 patch tokens)
        out = self.model.forward_features(x)
        
        # Extract CLS token as the feature representation
        features = out[:, 0, :]  # [B*D, 768]
        features = rearrange(features, '(b d) e -> b d e', b=B)  # [B, D, 768]
        return features

