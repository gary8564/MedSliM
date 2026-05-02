import logging
from pathlib import Path

import torch
import torch.nn as nn
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
        [B, C, W, H, D]  – grayscale volume (C=1), D = number of slices.

    Output:
        [B, D, embed_dim]  – one CLS-token embedding per slice.

    Normalization:
        Curia does NOT use fixed dataset statistics (no ImageNet mean/std).
        Per-slice z-score normalization — (x - mean_slice) / std_slice —
        is applied inside forward(), matching CuriaImageProcessor exactly.
        Do NOT apply any additional normalization in the dataloader.

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
        local_cache_dir: str = None,
    ):
        """
        Args:
            model_repo:      HuggingFace repo id (default: "raidium/curia").
            local_cache_dir: Directory to cache downloaded weights.
                             Defaults to HuggingFace's ~/.cache/huggingface/hub.
        """
        super().__init__()
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
            "Curia loaded: embed_dim=%d, params=%s",
            self.embed_dim,
            f"{sum(p.numel() for p in self.model.parameters()):,}",
        )

    @staticmethod
    def _zscore_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Per-slice z-score normalization: (x - mean) / std.

        Matches CuriaImageProcessor._zscore_per_image() exactly.
        Slices with near-zero std (blank slices) are left mean-subtracted only.

        Args:
            x: [(B*D), 1, H, W] grayscale slices.

        Returns:
            Normalized tensor, same shape.
        """
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std  = x.std(dim=(-2, -1), keepdim=True)
        std  = torch.where(std < eps, torch.ones_like(std), std)
        return (x - mean) / std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Volume tensor [B, C, W, H, D], C=1 (grayscale).

        Returns:
            features: [B, D, embed_dim]  – CLS token per slice.
        """
        x = x.swapaxes(2, -1)                              # [B, C, W, H, D] → [B, C, D, H, W]
        B, C, *_ = x.shape
        assert C == 1, f"Expected grayscale input (C=1), got C={C}"

        x = rearrange(x, "b c d h w -> (b d) c h w")      # [(B*D), 1, H, W]
        x = self._zscore_normalize(x)                      # per-slice z-score
        x = x.repeat(1, 3, 1, 1)                          # [(B*D), 3, H, W]  gray → RGB

        outputs = self.model(x, return_dict=True)

        # CLS token is at position 0 of last_hidden_state
        features = outputs.last_hidden_state[:, 0, :]      # [(B*D), embed_dim]
        features = rearrange(features, "(b d) e -> b d e", b=B)  # [B, D, embed_dim]
        return features


if __name__ == "__main__":
    from transformers import AutoModelForImageClassification
    import os

    model = AutoModelForImageClassification.from_pretrained(
        "raidium/curia",
        subfolder="kneeMRI",
        trust_remote_code=True,
        token=os.environ["HF_TOKEN"],
    )
    # inspect the attention module config
    print(model.config.attention_cfg)
    # inspect the trained attention module weights
    print(model.attention_module)
    print(model)