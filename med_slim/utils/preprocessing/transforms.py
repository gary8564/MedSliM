import torchio as tio 
from typing import Tuple, Optional

from med_slim.utils.preprocessing import (
    CropOrPad2D,
    CropOrPad3D,
    ZNormalization, 
    ImageOrSubjectToTensor, 
    ResizeInPlane,
    ResampleInPlane,
    AdaptivePreprocessing,
    EnsureSliceAxisLast,
)
from med_slim.utils.model_config import get_slice_encoder_config

def get_transforms(model_name: str,
                   num_slices: Optional[int] = None,
                   spatial_mode: str = 'resize', 
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
        spatial_mode: How to handle in-plane spatial dimensions:
            - 'resize': Scale to target size (recommended for 2.5D with pretrained 2D models)
            - 'resample': Resample maintaining physical spacing
            - 'crop': CropOrPad to target size
        random_rotate: Whether to random rotate the image
        random_center: Whether to random center the crop
        invert_intensity: Whether to invert the intensity of the image
        to_tensor: Whether to convert the TorchIO image to a tensor
    Returns:
        Tuple of (train_transform, val_transform)
    """
    # Get model-specific configuration
    config = get_slice_encoder_config(model_name)
    
    # Extract configuration parameters
    H_crop, W_crop = tuple(config["img_size"])
    D = num_slices
    means = list(config["image_mean"])
    stds = list(config["image_std"])
    
    # Build spatial transform based on mode
    if spatial_mode == 'resize':
        # ResizeInPlane handles W, H dimensions
        # If num_slices specified, crop or pad the depth dimension D as well
        if D is not None:
            spatial_transforms = [
                ResizeInPlane((W_crop, H_crop)),
                tio.CropOrPad((W_crop, H_crop, D), padding_mode='minimum'),
            ]
        else:
            spatial_transforms = [ResizeInPlane((W_crop, H_crop))]
    elif spatial_mode == 'resample':
        spatial_transforms = [ResampleInPlane((W_crop, H_crop), num_slices=D, image_interpolation="bspline")]
    elif spatial_mode == 'crop':
        if D is not None:
            spatial_transforms = [CropOrPad3D((W_crop, H_crop, D), random_center=random_center, padding_mode='minimum')]
        else:
            spatial_transforms = [CropOrPad2D((W_crop, H_crop), random_center=random_center, padding_mode='minimum')]
    else:
        raise ValueError(f"Unknown spatial_mode: {spatial_mode}")

    train_transform = tio.Compose([
                tio.ToCanonical(),      # Ensures consistent RAS+ orientation
                EnsureSliceAxisLast(),  # Ensures slice dimension is always at the last axis
                *spatial_transforms,    # Unpack spatial transforms
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
                EnsureSliceAxisLast(),
                *spatial_transforms,
                ZNormalization(per_channel=True, channelwise_precomputed_means=means, channelwise_precomputed_stds=stds, masking_method=lambda x: (x > x.min()) & (x < x.max())),
                ImageOrSubjectToTensor() if to_tensor else tio.Lambda(lambda x: x),
            ])
    return train_transform, val_transform 


def get_adaptive_transform( 
    model_name: str,
    num_slices: Optional[int] = None,
    to_tensor: bool = True,
) -> tio.Compose:
    """
    Get adaptive transform for a specific FM based on source resolution.
    """
    config = get_slice_encoder_config(model_name)
    H_target, W_target = tuple(config["img_size"])
    means = list(config["image_mean"])
    stds = list(config["image_std"])
    
    transforms_list = [
        tio.ToCanonical(),
        EnsureSliceAxisLast(),
        AdaptivePreprocessing(
            target_size=(W_target, H_target),
            num_slices=num_slices,
            padding_mode='minimum',
        ),
        ZNormalization(
            per_channel=True,
            channelwise_precomputed_means=means,
            channelwise_precomputed_stds=stds,
            masking_method=lambda x: (x > x.min()) & (x < x.max())
        ),
    ]
    
    if to_tensor:
        transforms_list.append(ImageOrSubjectToTensor())
    
    return tio.Compose(transforms_list)
