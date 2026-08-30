import logging
from pathlib import Path
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import AutoModel

from med_slim.logging.setup import init_logging

init_logging()
logger = logging.getLogger(__name__)


class CuriaFeatureExtractor(nn.Module):
    """
    Slice-level feature extractor backed by the Curia radiology foundation model
    (Dancette et al., 2025 - https://arxiv.org/abs/2509.06830).

    Curia is a DINOv2 ViT trained with self-supervised learning on 200M CT/MRI
    images from a single major hospital.  It is loaded via the HuggingFace
    transformers AutoModel interface with trust_remote_code=True.

    Input convention (shared with all MedSliM slice encoders):
        [B, C, W, H, D]  grayscale volume (C=1), D = number of slices.

    Output:
        [B, T, embed_dim], where T depends on token_mode:
        - "cls": one CLS-token embedding per slice, T = D.
        - "patch": flattened spatial patch tokens from every slice.
        - "cls_patch": CLS token followed by spatial patch tokens for every slice.

    Normalization:
        Curia does NOT use fixed dataset statistics (no ImageNet mean/std).
        Per-slice z-score normalization and optional CT air clipping, mirroring CuriaImageProcessor, are applied
        by MedSliM's preprocessing transforms.

    Notes:
        - The model weights are frozen by default (freeze=True in build_slice_encoder).
        - A HuggingFace token is required: accept the RAIL-M license at
          https://huggingface.co/raidium/curia and set HF_TOKEN in your environment.
        - embed_dim = 768 for Curia-B (ViT-B), 1024 for Curia-L (ViT-L).
    """

    DEFAULT_REPO = "raidium/curia"

    def __init__(
        self,
        model_repo: str = DEFAULT_REPO,
        local_cache_dir: Optional[str] = None,
        token_mode: Literal["cls", "patch", "cls_patch"] = "cls",
        spatial_pool_kernel_size: Optional[int] = None,
    ):
        """
        Args:
            model_repo: HuggingFace repo id (default: "raidium/curia").
            local_cache_dir: Directory to cache downloaded weights.
                             Defaults to HuggingFace's ~/.cache/huggingface/hub.
            token_mode: Which Curia tokens to return.
                        "cls" keeps MedSliM's existing slice-embedding convention.
                        "patch" and "cls_patch" expose Curia's spatial patch tokens
                        for attention pooling experiments.
            spatial_pool_kernel_size: Optional average-pooling kernel over the 2D patch grid before flattening patch tokens.
                                      Useful to reduce the sequence length of 512x512 Curia inputs (32x32 patches).
        """
        super().__init__()
        if token_mode not in {"cls", "patch", "cls_patch"}:
            raise ValueError(
                f"Unknown Curia token_mode '{token_mode}'. "
                "Choose from {'cls', 'patch', 'cls_patch'}."
            )
        if spatial_pool_kernel_size is not None and spatial_pool_kernel_size < 1:
            raise ValueError("spatial_pool_kernel_size must be >= 1 when provided.")

        self.token_mode = token_mode
        self.spatial_pool_kernel_size = spatial_pool_kernel_size

        if local_cache_dir is not None:
            local_cache_dir = Path(local_cache_dir)
            local_cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Loading Curia backbone from '%s' ...", model_repo)
        self.model = AutoModel.from_pretrained(
            model_repo,
            trust_remote_code=True,
            cache_dir=local_cache_dir,
        )
        self.embed_dim = self.model.config.hidden_size
        logger.info(
            "Curia loaded: embed_dim=%d, token_mode=%s, spatial_pool_kernel_size=%s, params=%s",
            self.embed_dim,
            self.token_mode,
            self.spatial_pool_kernel_size,
            f"{sum(p.numel() for p in self.model.parameters()):,}",
        )

    def _pool_patch_tokens(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        if self.spatial_pool_kernel_size is None or self.spatial_pool_kernel_size == 1:
            return patch_tokens

        num_patches = patch_tokens.shape[1]
        spatial_dim = int(num_patches ** 0.5)
        if spatial_dim * spatial_dim != num_patches:
            raise ValueError(
                f"Expected a square patch grid, got {num_patches} patch tokens."
            )

        patch_grid = rearrange(
            patch_tokens,
            "n (h w) e -> n e h w",
            h=spatial_dim,
            w=spatial_dim,
        )
        pooled = F.avg_pool2d(
            patch_grid,
            kernel_size=self.spatial_pool_kernel_size,
            stride=self.spatial_pool_kernel_size,
        )
        return rearrange(pooled, "n e h w -> n (h w) e")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Volume tensor [B, C, W, H, D], C=1 (grayscale).

        Returns:
            features: [B, T, embed_dim], with T determined by token_mode.
        """
        x = x.swapaxes(2, -1)                              # [B, C, W, H, D] → [B, C, D, H, W]
        B, C, *_ = x.shape
        assert C == 1, f"Expected grayscale input (C=1), got C={C}"

        x = rearrange(x, "b c d h w -> (b d) c h w")      # [(B*D), 1, H, W]

        outputs = self.model(pixel_values=x, return_dict=True)

        cls_tokens = outputs.last_hidden_state[:, 0:1, :]      # [(B*D), 1, embed_dim]
        patch_tokens = outputs.last_hidden_state[:, 1:, :]     # [(B*D), P, embed_dim]

        if self.token_mode == "cls":
            features = cls_tokens.squeeze(1)                   # [(B*D), embed_dim]
            return rearrange(features, "(b d) e -> b d e", b=B)

        patch_tokens = self._pool_patch_tokens(patch_tokens)
        if self.token_mode == "cls_patch":
            tokens = torch.cat([cls_tokens, patch_tokens], dim=1)
        else:
            tokens = patch_tokens

        features = rearrange(tokens, "(b d) p e -> b (d p) e", b=B)
        return features
