import torchio as tio 
import yaml
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

from med_slim.utils.preprocessing import CropOrPad, ZNormalization, ImageOrSubjectToTensor

def get_model_config(model_name: str) -> Dict[str, Any]:
    """
    Get model configurations from the YAML file.
    
    Args:
        model_name: Name of the pretrained model
        
    Returns:
        Dict containing model-specific configuration
    """
    current_dir = Path(__file__).parent
    configs_dir = current_dir.parent.parent / "configs"
    model_config_path = configs_dir / "model_config.yaml"
    
    if not model_config_path.exists():
        raise FileNotFoundError(f"Model configuration file not found: {model_config_path}")
    
    try:
        with open(model_config_path, 'r') as file:
            model_configs = yaml.safe_load(file)
        if "dinov2" in model_name: 
            model_name = "dinov2"
        return model_configs[model_name].copy()
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing model configuration file: {e}")
    except Exception as e:
        raise RuntimeError(f"Error loading model configuration file: {e}")

def get_transforms(model_name: str,
                   num_slices: int = 32,
                   resample: Optional[float] = None,
                   random_rotate: bool = False,
                   random_center: bool = False,
                   invert_intensity: bool = False,
                   noise: bool = False,
                   to_tensor: bool = False) -> Tuple[tio.Compose, tio.Compose]:
    """
    Define the transforms for data augmentation. 
    Uses model-specific configurations to ensure consistent image processing with the pretrained model.
    
    Args:
        model_name: Name of the pretrained model
        num_slices: Number of slices in the image
        resample: Resampling factor
        random_rotate: Whether to random rotate the image
        random_center: Whether to random center the crop
        invert_intensity: Whether to invert the intensity of 
        to_tensor: Whether to convert the torchioimage to a tensor
    Returns:
        Tuple of (train_transform, val_transform)
    """
    # Get model-specific configuration
    config = get_model_config(model_name)
    
    # Extract configuration parameters
    H_crop, W_crop = tuple(config["img_size"])
    D = num_slices
    means = list(config["image_mean"])
    stds = list(config["image_std"])

    train_transform = tio.Compose([
                tio.ToCanonical(),
                tio.Resample(resample) if resample is not None else tio.Lambda(lambda x: x),
                CropOrPad((W_crop, H_crop, D), random_center=random_center, padding_mode='minimum'), 
                ZNormalization(per_channel=True, channelwise_precomputed_means=means, channelwise_precomputed_stds=stds, masking_method=lambda x: (x > x.min()) & (x < x.max())),
                tio.OneOf({
                    tio.RandomAffine(scales=(0.9, 1.2), degrees=(-15, 15, -15, 15, 0, 90), translation=0, isotropic=True, default_pad_value='minimum'): 0.8,
                    tio.RandomElasticDeformation(): 0.2,
                }, p=0.75) if random_rotate else tio.Lambda(lambda x: x),
                tio.RandomFlip((0,1,2), p=0.5), 
                tio.Lambda(lambda x: -x, types_to_apply=[tio.INTENSITY], p=0.25) if invert_intensity else tio.Lambda(lambda x: x),
                tio.RandomNoise(std=(0.0, 0.25)) if noise else tio.Lambda(lambda x: x),
                ImageOrSubjectToTensor() if to_tensor else tio.Lambda(lambda x: x),
            ])

    val_transform = tio.Compose([
                tio.ToCanonical(),
                tio.Resample(resample) if resample is not None else tio.Lambda(lambda x: x),
                CropOrPad((W_crop, H_crop, D), random_center=random_center, padding_mode='minimum'), 
                ZNormalization(per_channel=True, channelwise_precomputed_means=means, channelwise_precomputed_stds=stds, masking_method=lambda x: (x > x.min()) & (x < x.max())),
                ImageOrSubjectToTensor() if to_tensor else tio.Lambda(lambda x: x),
            ])
    return train_transform, val_transform 