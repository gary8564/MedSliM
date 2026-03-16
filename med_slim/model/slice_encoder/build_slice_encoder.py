import os
from typing import Optional
import torch.nn as nn

from med_slim.utils.model_config import get_slice_encoder_config

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
        name: Name of the foundation model. ['ark', 'dinov2', 'dinov3', 'rad-dino', 'medsiglip', 'biomedclip', 'mri-core', 'medimageinsight']
        model_repo: Optional HF repo to override defaults for DINO/MedSigLIP/CLIP.
        checkpoint: Path to model checkpoint. Required for name='ark' and name='mri-core'.
        local_cache_dir: Local cache directory to store the model. If None, the default Hugging Face cache directory "~/.cache/huggingface/hub" will be used.
        freeze: If True, parameters are set to requires_grad=False.

    Returns:
        nn.Module: feature extractor model.
    """
    valid_names = ["ark", "dinov2", "dinov3", "rad-dino", "medsiglip", "biomedclip", "mri-core", "medimageinsight"]
    assert name in valid_names, f"Slice encoder '{name}' not supported. Choose from {valid_names}."
    config = get_slice_encoder_config(name)

    if config["name"] == "ark":
        from .ark import ArkFeatureExtractor
        ckpt = checkpoint or config["repo"]
        if ckpt is None or not os.path.exists(ckpt):
            raise FileNotFoundError(
                "Ark checkpoint not provided. "
                "Pass checkpoint=ckpt_path to build_slice_encoder."
            )
        model = ArkFeatureExtractor(model_checkpoint_path=ckpt, use_projector=freeze)

    elif config["name"] in ("dinov2", "dinov3", "rad-dino"):
        from .dino import DinoFeatureExtractor
        default_repo = config["repo"]
        repo = model_repo or default_repo
        model = DinoFeatureExtractor(model_repo=repo, local_cache_dir=local_cache_dir)

    elif config["name"] == "medsiglip":
        from .siglip import SigLipFeatureExtractor
        default_repo = config["repo"]
        repo = model_repo or default_repo
        model = SigLipFeatureExtractor(model_repo=repo, local_cache_dir=local_cache_dir)

    elif config["name"] == "biomedclip":
        from .clip import CLIPFeatureExtractor
        default_repo = config["repo"]
        repo = model_repo or default_repo
        model = CLIPFeatureExtractor(model_repo=repo, local_cache_dir=local_cache_dir)

    elif config["name"] == "mri-core":
        from .mri_core import MriCoreFeatureExtractor
        ckpt = checkpoint or config["repo"]
        if ckpt is None or not os.path.exists(ckpt):
            raise FileNotFoundError(
                "MRI-CORE checkpoint not provided. "
                "Download from https://github.com/mazurowski-lab/mri_foundation "
                "and pass checkpoint=ckpt_path to build_slice_encoder."
            )
        model = MriCoreFeatureExtractor(checkpoint_path=ckpt)

    elif config["name"] == "medimageinsight":
        from .medimageinsight import MedImageInsightFeatureExtractor
        model_dir = checkpoint or config["repo"]
        if model_dir is None or not os.path.isdir(model_dir):
            raise FileNotFoundError(
                "MedImageInsight model directory not provided or does not exist. "
                "Clone https://huggingface.co/lion-ai/MedImageInsights "
                "and pass checkpoint=<cloned_dir> to build_slice_encoder."
            )
        model = MedImageInsightFeatureExtractor(model_dir=model_dir)

    else:
        raise ValueError(f"Unknown slice encoder name: {name}")

    if freeze:
        for p in model.parameters():
            p.requires_grad_(False)

    return model


