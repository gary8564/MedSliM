import torchio as tio 
from typing import List, Tuple, Optional

from med_slim.utils.preprocessing import (
    CropOrPad2D,
    CropOrPad3D,
    CropEmptySlices,
    ZNormalization, 
    ImageOrSubjectToTensor, 
    ResizeInPlane,
    ResampleInPlane,
    AdaptivePreprocessing,
    EnsureSliceAxisLast,
    ClipIntensity,
    PerSliceZScore,
)
from med_slim.utils.model_config import get_slice_encoder_config

# Hounsfield unit (HU) window shared by the CT-specific slice encoders. MedDINOv3 and
# FlexiCT-2D both clamp CT to this range before their respective normalization
# (see their official inference demos).
CT_HU_MIN = -1000.0
CT_HU_MAX = 1000.0

def _build_intensity_transforms(
    model_name: str,
    modality: Optional[str],
    means: List[float],
    stds: List[float],
    masking_method,
) -> list:
    """
    Resolve the model-specific intensity pipeline from the model name.

    - `med-dinov3`: clamp CT to [-1000, 1000] then apply the fixed CT mean/std from config. 
    - `flexict-2d`: clamp CT to [-1000, 1000] then z-score each slice independently.
    - `curia`: optional CT air clipping followed by per-slice z-score, mirroring CuriaImageProcessor.
    - Other models: the standard fixed `image_mean`/`image_std` z-score.
    
    Note that `modality` only affects `curia` (CT enables air clipping, MRI skips it). 
    `med-dinov3` and `flexict-2d` are CT-specific, and `mri-core` is MRI-specific. 
    All other models ignore `modality` entirely.
    """
    if model_name == "med-dinov3":
        return [
            ClipIntensity(min_value=CT_HU_MIN, max_value=CT_HU_MAX),
            ZNormalization(
                per_channel=True,
                channelwise_precomputed_means=means,
                channelwise_precomputed_stds=stds,
            ),
        ]
    if model_name == "flexict-2d":
        return [
            ClipIntensity(min_value=CT_HU_MIN, max_value=CT_HU_MAX),
            PerSliceZScore(),
        ]
    if model_name == "curia":
        return [
            ClipIntensity(min_value=CT_HU_MIN, enabled=(modality == "ct")),
            PerSliceZScore(),
        ]
    return [
        ZNormalization(
            per_channel=True,
            channelwise_precomputed_means=means,
            channelwise_precomputed_stds=stds,
            masking_method=masking_method,
        ),
    ]

def get_transforms(model_name: str,
                   plane: str,
                   num_slices: Optional[int] = None,
                   spatial_mode: str = 'resize', 
                   random_rotate: bool = False,
                   random_center: bool = False,
                   invert_intensity: bool = False,
                   noise: bool = False,
                   crop_empty_slices: bool = False,
                   modality: Optional[str] = None,
                   to_tensor: bool = False) -> Tuple[tio.Compose, tio.Compose]:
    """
    Define the transforms for data augmentation. 
    Uses model-specific configurations to ensure consistent image processing with the pretrained model.
    
    Args:
        model_name: Name of the pretrained model
        plane: Acquisition plane ('axial', 'sagittal', 'coronal')
        num_slices: Number of slices in the image
        spatial_mode: How to handle in-plane spatial dimensions:
            - 'resize': Scale to target size (recommended for 2.5D with pretrained 2D models)
            - 'resample': Resample maintaining physical spacing
            - 'crop': CropOrPad to target size
        random_rotate: Whether to random rotate the image
        random_center: Whether to random center the crop
        invert_intensity: Whether to invert the intensity of the image
        noise: Whether to add random noise to the image
        crop_empty_slices: Whether to trim near-empty edge slices before spatial transforms
        modality: Acquisition modality ('ct' or 'mri'), or `None` when not
            applicable. Only affects Curia, where CT enables air clipping and MRI skips it. 
            Ignored by other models.
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
    masking_method = lambda x: (x > x.min()) & (x < x.max())
    intensity_transforms = _build_intensity_transforms(
        model_name, modality, means, stds, masking_method
    )
    
    # Build spatial transform based on mode
    if spatial_mode == 'resize':
        if D is not None:
            spatial_transforms = [
                ResizeInPlane((W_crop, H_crop), plane=plane),
                tio.CropOrPad((W_crop, H_crop, D), padding_mode='minimum'),
            ]
        else:
            spatial_transforms = [ResizeInPlane((W_crop, H_crop), plane=plane)]
    elif spatial_mode == 'resample':
        spatial_transforms = [ResampleInPlane((W_crop, H_crop), num_slices=D, image_interpolation="bspline", plane=plane)]
    elif spatial_mode == 'crop':
        if D is not None:
            spatial_transforms = [CropOrPad3D((W_crop, H_crop, D), random_center=random_center, padding_mode='minimum', plane=plane)]
        else:
            spatial_transforms = [CropOrPad2D((W_crop, H_crop), random_center=random_center, padding_mode='minimum', plane=plane)]
    else:
        raise ValueError(f"Unknown spatial_mode: {spatial_mode}")

    # Optional: trim near-empty edge slices before spatial transforms
    pre_spatial = [CropEmptySlices()] if crop_empty_slices else []

    train_transform = tio.Compose([
                tio.ToCanonical(),
                EnsureSliceAxisLast(plane=plane),
                *pre_spatial,           # Trim near-empty edge slices before spatial transforms
                *spatial_transforms,    # Unpack spatial transforms
                *intensity_transforms,  # Model-specific intensity transforms
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
                EnsureSliceAxisLast(plane=plane),
                *pre_spatial,
                *spatial_transforms,
                *intensity_transforms,  # Model-specific intensity transforms
                ImageOrSubjectToTensor() if to_tensor else tio.Lambda(lambda x: x),
            ])
    return train_transform, val_transform 


def get_adaptive_transform( 
    model_name: str,
    plane: str,
    num_slices: Optional[int] = None,
    crop_empty_slices: bool = False,
    modality: Optional[str] = None,
    to_tensor: bool = True,
) -> tio.Compose:
    """
    Get adaptive transform for a specific FM based on source resolution.

    Args:
        modality: Acquisition modality ('ct' or 'mri'), or `None` when not
            applicable. Only affects Curia, where CT enables air clipping and MRI skips it. 
            Ignored by other models.
    """
    config = get_slice_encoder_config(model_name)
    H_target, W_target = tuple(config["img_size"])
    means = list(config["image_mean"])
    stds = list(config["image_std"])
    masking_method = lambda x: (x > x.min()) & (x < x.max())
    intensity_transforms = _build_intensity_transforms(
        model_name, modality, means, stds, masking_method
    )
    
    transforms_list = [
        tio.ToCanonical(),
        EnsureSliceAxisLast(plane=plane),
    ]
    if crop_empty_slices:
        transforms_list.append(CropEmptySlices())
    transforms_list.append(
        AdaptivePreprocessing(
            target_size=(W_target, H_target),
            num_slices=num_slices,
            padding_mode='minimum',
            plane=plane,
        ),
    )
    transforms_list.extend(intensity_transforms)
    
    if to_tensor:
        transforms_list.append(ImageOrSubjectToTensor())
    
    return tio.Compose(transforms_list)
