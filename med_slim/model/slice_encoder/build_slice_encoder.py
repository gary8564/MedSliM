import os
from typing import Optional
import torch.nn as nn

from .ark import ArkFeatureExtractor
from .dino import DinoFeatureExtractor
from .siglip import SigLipFeatureExtractor
from .clip import CLIPFeatureExtractor
from med_slim.utils.preprocessing import get_model_config

def build_slice_encoder(
    name: str,
    *,
    model_repo: Optional[str] = None,
    checkpoint: Optional[str] = None,
    local_cache_dir: Optional[str] = None,
    freeze: bool = True,
) -> nn.Module:
    """
    Factory to construct a pretrained 2D slice feature extractor.

    Args:
        name: Name of the foundation model. ['ark', 'dinov2', 'dinov3', 'rad-dino', 'medsiglip', 'biomedclip']
        model_repo: Optional HF repo to override defaults for DINO/MedSigLIP/CLIP.
        checkpoint: Path to Ark checkpoint. Required for name='ark'.
        local_cache_dir: Local cache directory to store the model. If None, the default Hugging Face cache directory "~/.cache/huggingface/hub" will be used.
        freeze: If True, parameters are set to requires_grad=False.

    Returns:
        nn.Module: feature extractor model.
    """
    assert name in ["ark", "dinov2", "dinov3", "rad-dino", "medsiglip", "biomedclip"], "Slice encoder not supported."
    config = get_model_config(name)

    if config["name"] == "ark":
        ckpt = checkpoint or config["repo"]
        if ckpt is None or not os.path.exists(ckpt):
            raise FileNotFoundError(
                "Ark checkpoint not provided. "
                "Pass checkpoint=ckpt_path to build_slice_encoder."
            )
        model = ArkFeatureExtractor(model_checkpoint_path=ckpt, use_projector=freeze)

    elif config["name"] in ("dinov2", "dinov3", "rad-dino"):
        default_repo = config["repo"]
        repo = model_repo or default_repo
        model = DinoFeatureExtractor(model_repo=repo, local_cache_dir=local_cache_dir)

    elif config["name"] == "medsiglip":
        default_repo = config["repo"]
        repo = model_repo or default_repo
        model = SigLipFeatureExtractor(model_repo=repo, local_cache_dir=local_cache_dir)

    elif config["name"] == "biomedclip":
        default_repo = config["repo"]
        repo = model_repo or default_repo
        model = CLIPFeatureExtractor(model_repo=repo, local_cache_dir=local_cache_dir)

    else:
        raise ValueError(f"Unknown slice encoder name: {name}")

    if freeze:
        for p in model.parameters():
            p.requires_grad_(False)

    return model


