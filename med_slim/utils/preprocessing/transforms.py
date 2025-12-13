import torchio as tio 
from typing import Tuple, Optional

from med_slim.utils.preprocessing import CropOrPad, CropOrPad2D, ZNormalization, ImageOrSubjectToTensor, get_model_config

def get_transforms(model_name: str,
                   num_slices: Optional[int] = None,
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
                CropOrPad((W_crop, H_crop, D), random_center=random_center, padding_mode='minimum') if D is not None else CropOrPad2D((W_crop, H_crop), random_center=random_center, padding_mode='minimum'), 
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
                CropOrPad((W_crop, H_crop, D), random_center=random_center, padding_mode='minimum') if D is not None else CropOrPad2D((W_crop, H_crop), random_center=random_center, padding_mode='minimum'), 
                ZNormalization(per_channel=True, channelwise_precomputed_means=means, channelwise_precomputed_stds=stds, masking_method=lambda x: (x > x.min()) & (x < x.max())),
                ImageOrSubjectToTensor() if to_tensor else tio.Lambda(lambda x: x),
            ])
    return train_transform, val_transform 