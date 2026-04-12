import os
import sys
import torch
import torch.nn as nn
import logging
from einops import rearrange

from med_slim.logging.setup import init_logging
init_logging()
logger = logging.getLogger(__name__)


def _load_unicl_model(model_dir: str, device: str = "cpu"):
    """
    Load the MedImageInsight UniCL model from a locally cloned
    lion-ai/MedImageInsights repository.

    Args:
        model_dir: Absolute path to the cloned ``lion-ai/MedImageInsights``
                   huggingface repository.
        device: Target device, e.g. "cpu" or "cuda".

    Returns:
        The loaded ``UniCLModel`` (nn.Module) on the requested device.
    """
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)

    from MedImageInsight.UniCLModel import build_unicl_model
    from MedImageInsight.Utils.Arguments import load_opt_from_config_files

    config_path = os.path.join(model_dir, "2024.09.27", "config.yaml")
    config = load_opt_from_config_files([config_path])

    config["UNICL_MODEL"]["PRETRAINED"] = os.path.join(
        model_dir, "2024.09.27", "vision_model", "medimageinsigt-v1.0.0.pt"
    )
    config["LANG_ENCODER"]["PRETRAINED_TOKENIZER"] = os.path.join(
        model_dir, "2024.09.27", "language_model", "clip_tokenizer_4.16.2"
    )

    model = build_unicl_model(config)
    model.to(device)
    logger.info(
        "Loaded MedImageInsight UniCL model from %s (device=%s)",
        model_dir, device,
    )
    return model


class MedImageInsightFeatureExtractor(nn.Module):
    """
    Slice feature extractor backed by MedImageInsight (UniCL / DaViT).
    """

    def __init__(
        self,
        model_dir: str,
        device: str = "cpu",
        chunk_size: int = 4,
    ):
        """
        Args:
            model_dir: Path to the cloned lion-ai/MedImageInsights repo.
            device:    Device used for loading weights (parameters are later
                       moved by the caller / build_slice_encoder).
            chunk_size: Number of slices forwarded at once to avoid OOM on
                        the DaViT backbone.
        """
        super().__init__()
        self.model = _load_unicl_model(model_dir, device=device)
        self.embed_dim: int = self.model.image_projection.shape[1]  # 1024
        self.chunk_size = chunk_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, W, H, D] grayscale medical volume.

        Returns:
            [B, D, embed_dim] per-slice feature vectors.
        """
        x = x.swapaxes(2, -1)  # [B, C, W, H, D] -> [B, C, D, H, W]
        B, C, *_ = x.shape
        assert C == 1, "MRI/CT slices should be grayscale (C = 1)"

        # Flatten batch and depth -> (B*D, 1, H, W) then replicate to RGB
        x = rearrange(x, "b c d h w -> (b d) c h w")
        x = x.repeat(1, 3, 1, 1)  # [B*D, 1, H, W] -> [B*D, 3, H, W]

        total_slices = x.shape[0]
        outputs = []
        for start in range(0, total_slices, self.chunk_size):
            end = min(start + self.chunk_size, total_slices)
            chunk = x[start:end]
            with torch.no_grad():
                feat = self.model.encode_image(chunk, norm=True)
            outputs.append(feat)
            del chunk

        features = torch.cat(outputs, dim=0)  # [B*D, embed_dim]
        features = rearrange(features, "(b d) e -> b d e", b=B)
        return features
