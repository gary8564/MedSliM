import os
from typing import Optional
import torch.nn as nn

from .ark import ArkFeatureExtractor
from .dino import DinoFeatureExtractor
from .siglip import SigLipFeatureExtractor
from .clip import CLIPFeatureExtractor


def build_slice_encoder(
    name: str,
    *,
    model_repo: Optional[str] = None,
    checkpoint: Optional[str] = None,
    freeze: bool = True,
) -> nn.Module:
    """
    Factory to construct a pretrained 2D slice feature extractor.

    Args:
        name: Name of the foundation model. ['ark', 'dinov2', 'rad-dino', 'medsiglip', 'biomedclip']
        model_repo: Optional HF repo to override defaults for DINO/MedSigLIP/CLIP.
        checkpoint: Path to Ark checkpoint. Required for name='ark'.
        freeze: If True, parameters are set to requires_grad=False.

    Returns:
        nn.Module: feature extractor model.
    """
    key = name.lower()

    if key == "ark":
        ckpt = checkpoint
        if ckpt is None or not os.path.exists(ckpt):
            raise FileNotFoundError(
                "Ark checkpoint not provided. "
                "Pass checkpoint=ckpt_path to build_slice_encoder."
            )
        model = ArkFeatureExtractor(model_checkpoint_path=ckpt, use_projector=freeze)

    elif key in ("dinov2", "rad-dino"):
        default_repo = "facebook/dinov2-large" if key != "rad-dino" else "microsoft/rad-dino"
        repo = model_repo or default_repo
        model = DinoFeatureExtractor(model_repo=repo)

    elif key in ("medsiglip"):
        model = SigLipFeatureExtractor(model_repo="google/medsiglip-448")

    elif key in ("biomedclip"):
        model = CLIPFeatureExtractor(model_repo="microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")

    else:
        raise ValueError(f"Unknown slice encoder name: {name}")

    if freeze:
        for p in model.parameters():
            p.requires_grad_(False)

    return model


