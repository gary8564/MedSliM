"""
Adapted from https://github.com/mueller-franzes/MST/blob/main/mst/data/datasets/augmentations/augmentations_3d.py
Müller-Franzes, Gustav, Firas Khader, Robert Siepmann, Tianyu Han, Jakob Nikolas Kather, Sven Nebelung and Daniel Truhn. 
"Medical Slice Transformer: Improved Diagnosis and Explainability on 3D Medical Images with DINOv2."
ArXiv abs/2411.15802 (2024).
"""
import torchio as tio 
import torch.nn.functional as F
import torch 
import nibabel as nib 
import numpy as np
from typing import Iterable, Tuple, Union, List, Optional, Sequence, Dict, Callable
from numbers import Number
from torchio.transforms.transform import TypeMaskingMethod 
from torchio import Subject, Image

TypeRangeFloat = Tuple[float, float]  # type: ignore
TypeTripletInt = Union[int, Tuple[int, int, int], Sequence[int]]  # type: ignore


def _slice_axis_from_subject(subject: tio.Subject) -> int:
    """
    Identify slice (through-plane) axis by largest spacing (thick slices)
    or smallest dimension.

    For RAS+ oriented images (after ToCanonical):
    - Axial scans: slice axis = 2 (I-S direction)
    - Sagittal scans: slice axis = 0 (R-L direction)  
    - Coronal scans: slice axis = 1 (A-P direction)
    """
    current_spacing = np.array(subject.spacing)
    # Use spacing to detect slice axis (slice thickness = through-plane direction)
    if np.std(current_spacing) > 1e-6:
        return int(np.argmax(current_spacing))
    return int(np.argmin(subject.spatial_shape))


def _permute_slice_to_last(subject: tio.Subject, slice_axis: int) -> tio.Subject:
    """
    Permute spatial axes so slice dimension moves to last position (D).
    Ensures ImageOrSubjectToTensor's swapaxes(1,-1) produces (C, D, H, W) with D=slices.
    Also updates the affine matrix to maintain correct physical coordinates.
    """
    if slice_axis == 2:
        return subject
    
    # Build permutation
    perm = [i for i in range(3) if i != slice_axis] + [slice_axis]
    data_perm = (0,) + tuple(p + 1 for p in perm)
    
    for image in subject.get_images_dict().values():
        data = image.data
        if isinstance(data, torch.Tensor):
            permuted = data.permute(data_perm)
            image.set_data(permuted)
        
        # Update affine: reorder columns to match new axis order
        aff = np.array(image.affine, copy=True)
        new_aff = np.eye(4)
        new_aff[:3, :3] = aff[:3, perm]  # Reorder columns
        new_aff[:3, 3] = aff[:3, 3]      # Keep translation
        image.affine = new_aff
    
    return subject


class EnsureSliceAxisLast(tio.Transform):
    """
    Ensure the slice (through-plane) axis is always at position 2 (last spatial dimension).
    Should be applied after ToCanonical() to ensure consistent axis ordering
    regardless of the original acquisition plane (axial, sagittal, coronal).
    Crucial for applying other transforms that require a specific axis ordering
    
    After this transform:
    - Image shape is (C, W, H, D)
    - Spacing order matches the data order
    - The affine matrix is updated to maintain physical coordinates
    """
    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        slice_axis = _slice_axis_from_subject(subject)
        return _permute_slice_to_last(subject, slice_axis)

class SubjectToTensor(object):
    """Transforms TorchIO Subjects into a Python dict and changes axes order from TorchIO to Torch"""
    def __call__(self, subject: Subject):
        return {key: val.data.swapaxes(1,-1) if isinstance(val, Image) else val for key,val in subject.items()}

class ImageToTensor(object):
    """Transforms TorchIO Image into a Numpy/Torch Tensor and changes axes order from TorchIO [B, C, W, H, D] to Torch [B, C, D, H, W]"""
    def __call__(self, image: Image):
        return image.data.swapaxes(1,-1)

class ImageOrSubjectToTensor(object):
    """Depending on the input, it will either run SubjectToTensor or ImageToTensor"""
    def __call__(self, input: Union[Image, Subject]):
        if isinstance(input, Subject):
            return {key: val.data.swapaxes(1,-1) if isinstance(val, Image) else val  for key,val in input.items()}
        else:
            return input.data.swapaxes(1,-1)

class ZNormalization(tio.ZNormalization):
    """
    Apply Z-score normalization to a given input tensor.
    Compared to the official TorchIO ZNormalization, which does one global z-norm per image, this class extends the option to:
    (1) Adds robustness via percentile clipping before computing mean/std.
    (2) Adds control over the normalization granularity: per-channel vs global.
    """
    def __init__(
        self,
        percentiles: TypeRangeFloat=(0, 100),
        per_channel: bool=True,
        channelwise_precomputed_means: Optional[List[float]]=None,
        channelwise_precomputed_stds: Optional[List[float]]=None,
        masking_method: TypeMaskingMethod=None,
        **kwargs
    ):
        super().__init__(masking_method=masking_method, **kwargs)
        self.args_names = ['masking_method', 'percentiles', 'per_channel', 'channelwise_precomputed_means', 'channelwise_precomputed_stds']
        self.percentiles = percentiles
        self.per_channel = per_channel
        self.channelwise_precomputed_means = channelwise_precomputed_means 
        self.channelwise_precomputed_stds = channelwise_precomputed_stds 
        
    def _parse_per_channel(self, channels: int) -> List[Tuple[int]]:
        if self.per_channel:
            return [(ch,) for ch in range(channels)]
        else:
            return [tuple(ch for ch in range(channels))] 
        
    def apply_normalization(
        self,
        subject: Subject,
        image_name: str,
        mask: torch.Tensor,
    ) -> None:
        image = subject[image_name]
        per_channel = self._parse_per_channel(image.shape[0])
        means = self.channelwise_precomputed_means if self.channelwise_precomputed_means is not None else [None] * image.shape[0]
        stds = self.channelwise_precomputed_stds if self.channelwise_precomputed_stds is not None else [None] * image.shape[0]
        # Normalize each channel independently
        image.set_data(
            torch.cat([
                self._znorm(
                    image.data[chs,],
                    mask[chs,],
                    image_name,
                    image.path,
                    means[chs[0]],
                    stds[chs[0]],
                )
                for chs in per_channel
            ])
        )

    def _znorm(self, image_data, mask, image_name, image_path, mean, std):
        # torch.quantile() fails for tensors > ~16M elements, use numpy for large volumes
        masked_values = image_data.masked_select(mask).float()
        if masked_values.numel() > 10_000_000:
            cutoff = np.quantile(masked_values.cpu().numpy(), np.array(self.percentiles) / 100.0)
            cutoff = torch.tensor(cutoff, dtype=image_data.dtype, device=image_data.device)
        else:
            cutoff = torch.quantile(masked_values, torch.tensor(self.percentiles) / 100.0)
        torch.clamp(image_data, *cutoff.to(image_data.dtype).tolist(), out=image_data)
        if mean is not None and std is not None:
            if std == 0:
                standardized = None
            else:
                standardized = (image_data - mean) / std
        else:
            standardized = self.znorm(image_data, mask)
        if standardized is None:
            message = (
                'Standard deviation is 0 for masked values'
                f' in image "{image_name}" ({image_path})'
            )
            raise RuntimeError(message)
        return standardized

class EnsureShapeMultiple(tio.EnsureShapeMultiple):
    """
    Ensure that all values in the image shape are divisible by :math:`n`.
    Some convolutional neural network architectures need that the size of the input across all spatial dimensions is a power of :math:`2`.
    Extended version adds option 'padding_mode' to specify the padding mode.
    """
    def __init__(self, target_multiple, *, method: str = 'pad', padding_mode=0, **kwargs):
        super().__init__(target_multiple, method=method, **kwargs)
        self.padding_mode = padding_mode 
    
    def apply_transform(self, subject: Subject) -> Subject:
        source_shape = np.array(subject.spatial_shape, np.uint16)
        function: Callable = np.floor if self.method == 'crop' else np.ceil  # type: ignore[assignment]  # noqa: B950
        integer_ratio = function(source_shape / self.target_multiple)
        target_shape = integer_ratio * self.target_multiple
        target_shape = np.maximum(target_shape, 1)
        transform = tio.CropOrPad(target_shape.astype(int), padding_mode=self.padding_mode, **self.get_base_args()) 
        subject = transform(subject)  # type: ignore[assignment]
        return subject

class CropOrPad(tio.CropOrPad):
    """
    Unlike official TorchIO CropOrPad transform, which only supports fully deterministic crop/pad centered either on the volume center or on a mask center.
    The extended version adds a 'random_center' option to randomly center the crop or pad if no mask is set otherwise only random padding.
    """

    def __init__(
        self,
        target_shape: Union[int, TypeTripletInt, None] = None,
        padding_mode: Union[str, float] = 0,
        mask_name: Optional[str] = None,
        labels: Optional[Sequence[int]] = None,
        only_crop: bool = False,
        only_pad: bool = False,
        random_center: bool = False,
        **kwargs,
    ):
        super().__init__(
            target_shape=target_shape,
            padding_mode=padding_mode,
            mask_name=mask_name,
            labels=labels,
            only_crop=only_crop,
            only_pad=only_pad,
            **kwargs
        )
        self.random_center = random_center
    
    def _get_six_bounds_parameters(self, parameters: np.ndarray):
        result = []
        for number in parameters:
            if self.random_center:
                ini = np.random.randint(low=0, high=number+1)
            else:
                ini = int(np.ceil(number / 2)) # Center the crop/pad
            fin = number - ini
            result.extend([ini, fin])
        return tuple(result) # Returns a tuple of 6 bounds parameters: (i1, i2, j1, j2, k1, k2)
    
    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        subject.check_consistent_space()
        padding_params, cropping_params = self.compute_crop_or_pad(subject)
        padding_kwargs = {'padding_mode': self.padding_mode}
        if padding_params is not None and not self.only_crop:
            if self.random_center:
                random_padding_params = []
                for i in range(0, len(padding_params), 2):
                    s = padding_params[i] + padding_params[i + 1]
                    r = np.random.randint(0, s+1)
                    random_padding_params.extend([r, s - r])
                padding_params = random_padding_params
            pad = tio.Pad(padding_params, **self.get_base_args(), **padding_kwargs)
            subject = pad(subject)  # type: ignore[assignment]
        if cropping_params is not None and not self.only_pad:
            crop = tio.Crop(cropping_params, **self.get_base_args())
            subject = crop(subject)  # type: ignore[assignment]
        return subject


class CropOrPad3D(tio.Transform):
    """
    Crop or pad all three dimensions with automatic slice axis detection.
    
    Handles ToCanonical() reorientation by detecting the slice axis from spacing
    and mapping the target (W, H, D) to the correct physical axes.
    
    Args:
        target_shape: Target shape as (W_inplane, H_inplane, D_slices)
        padding_mode: Padding mode (see tio.Pad for options)
        random_center: randomly center the crop/pad if set to True; otherwise, center crop or pad
    """

    def __init__(
        self,
        target_shape: Tuple[int, int, int],
        padding_mode: Union[str, float] = 0,
        random_center: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_w, self.target_h, self.target_d = target_shape
        self.padding_mode = padding_mode
        self.random_center = random_center

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        subject.check_consistent_space()
        
        # Detect slice axis 
        slice_axis = _slice_axis_from_subject(subject)
        in_plane_axes = [i for i in range(3) if i != slice_axis]
        
        # Build target shape mapping: user's (W, H, D) -> actual axes
        actual_target = [0, 0, 0]
        actual_target[in_plane_axes[0]] = self.target_w
        actual_target[in_plane_axes[1]] = self.target_h
        actual_target[slice_axis] = self.target_d
        
        # Apply CropOrPad with correctly mapped target shape
        crop_or_pad = CropOrPad(
            target_shape=tuple(actual_target),
            padding_mode=self.padding_mode,
            random_center=self.random_center,
        )
        subject = crop_or_pad(subject)
        
        # Permute so slice dimension is last (D) for ImageOrSubjectToTensor compatibility
        subject = _permute_slice_to_last(subject, slice_axis)
        
        return subject


class CropOrPad2D(tio.Transform):
    """
    Crop or pad only the in-plane dimensions, leaving the slice (through-plane) dimension unchanged.
    
    Handles ToCanonical() reorientation by detecting the slice axis from spacing.
    
    Args:
        target_shape_2d: Target shape for in-plane (W, H) dimensions
        padding_mode: Padding mode (see tio.Pad for options)
        random_center: If True, randomly center the crop/pad; if False, center crop/pad
    """

    def __init__(
        self,
        target_shape_2d: Union[int, Tuple[int, int], None] = None,
        padding_mode: Union[str, float] = 0,
        random_center: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if isinstance(target_shape_2d, int):
            target_shape_2d = (target_shape_2d, target_shape_2d)
        self.target_size = target_shape_2d
        self.padding_mode = padding_mode
        self.random_center = random_center

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        subject.check_consistent_space()
        
        # Get current spatial shape
        current_shape = np.array(subject.spatial_shape)
        
        # Detect slice axis 
        slice_axis = _slice_axis_from_subject(subject)
        
        # Build target shape: crop in-plane dims to (target_w, target_h), keep slice dim unchanged
        target_shape = np.array(current_shape, copy=True)
        in_plane_axes = [i for i in range(3) if i != slice_axis]
        target_shape[in_plane_axes[0]] = self.target_size[0]
        target_shape[in_plane_axes[1]] = self.target_size[1]
        target_shape = tuple(target_shape.tolist())
        
        # Use CropOrPad with the computed target shape
        crop_or_pad = CropOrPad(
            target_shape=tuple(target_shape),
            padding_mode=self.padding_mode,
            random_center=self.random_center,
        )
        subject = crop_or_pad(subject)
        
        # Permute so slice dimension is last (D) for ImageOrSubjectToTensor compatibility
        subject = _permute_slice_to_last(subject, slice_axis)
        
        return subject
    
class ResampleInPlane(tio.Transform):
    """
    Resample the in-plane dimensions (W, H) to a target size while optionally
    adjusting the number of slices (D).
    
    This preserves all anatomical information by interpolation rather than cropping.
    
    Args:
        target_shape_2d: Target (W, H) size for in-plane dimensions
        num_slices: Optional target number of slices. If None, keeps original slice count.
        image_interpolation: Interpolation mode for image data ('linear', 'nearest', 'bspline')
    """
    
    def __init__(
        self,
        target_shape_2d: Tuple[int, int],
        num_slices: Optional[int] = None,
        image_interpolation: str = 'linear',
        **kwargs,
    ):
        super().__init__(**kwargs)
        if isinstance(target_shape_2d, int):
            target_shape_2d = (target_shape_2d, target_shape_2d)
        self.target_w, self.target_h = target_shape_2d
        self.num_slices = num_slices
        self.image_interpolation = image_interpolation
    
    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        # Get current shape and spacing
        current_shape = np.array(subject.spatial_shape)  # (W, H, D)
        current_spacing = np.array(subject.spacing)  # (sw, sh, sd)
        # Find the slice axis
        slice_axis = _slice_axis_from_subject(subject)
        in_plane_axes = [i for i in range(3) if i != slice_axis]
        
        # Calculate new spacing to achieve target in-plane shape
        # new_spacing = current_spacing * (current_shape / target_shape)
        new_spacing = current_spacing.copy()
        
        # For in-plane dimensions
        target_inplane = [self.target_w, self.target_h]
        for idx, axis in enumerate(in_plane_axes):
            new_spacing[axis] = current_spacing[axis] * current_shape[axis] / target_inplane[idx]
        
        # For slice dimension
        if self.num_slices is not None:
            new_spacing[slice_axis] = current_spacing[slice_axis] * current_shape[slice_axis] / self.num_slices
        
        # Apply resampling
        resample = tio.Resample(target=tuple(new_spacing), image_interpolation=self.image_interpolation)
        subject = resample(subject)
        
        # Due to floating point precision, we might be off by 1 voxel
        # Use CropOrPad to ensure exact target shape
        target_shape = [0, 0, 0]
        target_shape[in_plane_axes[0]] = self.target_w
        target_shape[in_plane_axes[1]] = self.target_h
        target_shape[slice_axis] = self.num_slices if self.num_slices else current_shape[slice_axis]
        
        crop_or_pad = tio.CropOrPad(target_shape=tuple(target_shape), padding_mode='minimum')
        subject = crop_or_pad(subject)
        
        # Permute so slice dimension is last for ImageOrSubjectToTensor compatibility
        subject = _permute_slice_to_last(subject, slice_axis)
        
        return subject
    
    
class ResizeInPlane(tio.Transform):
    """
    Resize only the in-plane dimensions (W, H) while keeping the slice dimension unchanged.
        
    Unlike Resample (which maintains physical spacing) or CropOrPad (which loses/adds data),
    this preserves all anatomical information by scaling.
    
    Args:
        target_size: Target (W, H) size for in-plane dimensions
        mode: Interpolation mode ('bilinear', 'bicubic', 'nearest')
        align_corners: If True, align corner pixels. Only used for bilinear/bicubic.
    
    Note:
        - This changes the effective pixel spacing (mm/pixel) in the in-plane dimensions
        - The slice (through-plane) dimension and spacing remain unchanged
    """
    
    def __init__(
        self,
        target_size: Union[int, Tuple[int, int]],
        mode: str = 'bilinear',
        align_corners: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        self.target_h, self.target_w = target_size  # Follow (H, W) convention for F.interpolate
        self.mode = mode
        self.align_corners = align_corners if mode in ('bilinear', 'bicubic') else None
    
    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        # Permute so slice dimension is last first (resolves ToCanonical axis reordering)
        # Find the slice axis
        slice_axis = _slice_axis_from_subject(subject)
        subject = _permute_slice_to_last(subject, slice_axis)
        
        for image_name, image in subject.get_images_dict().items():
            # Get image data: (C, W, H, D) in TorchIO convention
            data = image.data  # torch.Tensor
            C, W, H, D = data.shape
            
            # Reshape for F.interpolate: need (N, C, H, W) format
            # Process each slice independently
            # Reshape from (C, W, H, D) to (D, C, H, W) - treat slices as batch
            data = data.permute(3, 0, 2, 1)  # (D, C, H, W)
            
            # Resize in-plane
            if self.align_corners is not None:
                resized = F.interpolate(
                    data.float(),
                    size=(self.target_h, self.target_w),
                    mode=self.mode,
                    align_corners=self.align_corners,
                )
            else:
                resized = F.interpolate(
                    data.float(),
                    size=(self.target_h, self.target_w),
                    mode=self.mode,
                )
            
            # Reshape back to TorchIO convention: (D, C, H, W) -> (C, W, H, D)
            resized = resized.permute(1, 3, 2, 0)  # (C, W, H, D)
            
            # Update affine matrix to reflect new spacing
            # New in-plane spacing = old_spacing * (old_size / new_size)
            old_affine = image.affine.copy()
            new_affine = old_affine.copy()
            
            # Scale factors for in-plane dimensions
            scale_w = W / self.target_w
            scale_h = H / self.target_h
            
            # Update the affine matrix scaling (first 3x3 block contains rotation and scaling)
            # For simplicity, we scale the voxel sizes in the affine
            new_affine[0, 0] *= scale_w  # X spacing
            new_affine[1, 1] *= scale_h  # Y spacing
            # Z spacing (slice) unchanged
            
            # Create new image with updated data and affine
            new_image = tio.ScalarImage(tensor=resized.to(data.dtype), affine=new_affine)
            subject[image_name] = new_image
        
        return subject

class AdaptivePreprocessing(tio.Transform):
    """
    Adaptive preprocessing that chooses optimal strategy based on 
    source resolution and target FM requirements.
    
    Strategy:
    - Upsampling needed (target > source): Use Resample (no choice)
    - Minor downsampling (0.7 < scale < 1.0): Use CropOrPad (preserve native resolution)
    - Major downsampling (scale < 0.7): Use Resample then CropOrPad (preserve anatomy + sharpness)
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int],
        num_slices: Optional[int] = None,
        padding_mode: str = 'minimum',
        resample_interpolation: str = 'linear',
        resize_mode: str = 'area', 
        **kwargs
    ):
        super().__init__(**kwargs)
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        self.target_w, self.target_h = target_size
        self.num_slices = num_slices
        self.padding_mode = padding_mode
        self.resample_interpolation = resample_interpolation
        self.resize_mode = resize_mode
    
    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        # Get current shape
        current_shape = np.array(subject.spatial_shape)  # (W, H, D)
        current_spacing = np.array(subject.spacing)
        
        # Detect slice axis
        slice_axis = int(np.argmin(current_shape))
        in_plane_axes = [i for i in range(3) if i != slice_axis]
        
        W_source = current_shape[in_plane_axes[0]]
        H_source = current_shape[in_plane_axes[1]]
        D_source = current_shape[slice_axis]
        
        # Calculate scale factors
        scale_w = self.target_w / W_source
        scale_h = self.target_h / H_source
        avg_scale = (scale_w + scale_h) / 2
        
        target_d = self.num_slices if self.num_slices else D_source
        
        # Choose strategy based on scale
        if avg_scale > 1.0:
            # UPSAMPLING: Resample (interpolation required)
            strategy = 'resample'
        elif avg_scale > 0.7:
            # MINOR DOWNSAMPLING: CropOrPad preserves native resolution
            strategy = 'crop'
        else:
            # MAJOR DOWNSAMPLING: Resample first, then CropOrPad
            strategy = 'resample_then_crop'
        
        # Build target shape
        target_shape = [0, 0, 0]
        target_shape[in_plane_axes[0]] = self.target_w
        target_shape[in_plane_axes[1]] = self.target_h
        target_shape[slice_axis] = target_d
        
        # Apply strategy
        if strategy == 'crop':
            # Direct CropOrPad
            transform = tio.CropOrPad(
                target_shape=tuple(target_shape),
                padding_mode=self.padding_mode
            )
            subject = transform(subject)
            
        elif strategy == 'resample':
            # Calculate target spacing for resampling
            new_spacing = current_spacing * (current_shape / np.array(target_shape))
            
            resample = tio.Resample(
                target=tuple(new_spacing),
                image_interpolation=self.resample_interpolation
            )
            subject = resample(subject)
            
            # Ensure exact shape (floating point rounding)
            crop_or_pad = tio.CropOrPad(
                target_shape=tuple(target_shape),
                padding_mode=self.padding_mode
            )
            subject = crop_or_pad(subject)
            
        else: 
            # Step 1: Resample to intermediate size (10-15% larger than target)
            intermediate_scale = 1.15
            intermediate_shape = [0, 0, 0]
            intermediate_shape[in_plane_axes[0]] = int(self.target_w * intermediate_scale)
            intermediate_shape[in_plane_axes[1]] = int(self.target_h * intermediate_scale)
            intermediate_shape[slice_axis] = target_d
            
            intermediate_spacing = current_spacing * (current_shape / np.array(intermediate_shape))
            
            resample = tio.Resample(
                target=tuple(intermediate_spacing),
                image_interpolation=self.resample_interpolation
            )
            subject = resample(subject)
            
            # Step 2: CropOrPad to exact target (preserves sharpness)
            crop_or_pad = tio.CropOrPad(
                target_shape=tuple(target_shape),
                padding_mode=self.padding_mode
            )
            subject = crop_or_pad(subject)
        
        return subject