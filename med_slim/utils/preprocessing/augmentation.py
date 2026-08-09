"""
Adapted from https://github.com/mueller-franzes/MST/blob/main/mst/data/datasets/augmentations/augmentations_3d.py
Müller-Franzes, Gustav, Firas Khader, Robert Siepmann, Tianyu Han, Jakob Nikolas Kather, Sven Nebelung and Daniel Truhn. 
"Medical Slice Transformer: Improved Diagnosis and Explainability on 3D Medical Images with DINOv2."
ArXiv abs/2411.15802 (2024).
"""
import torchio as tio 
import torch.nn.functional as F
import torch 
import numpy as np
from typing import Tuple, Union, List, Optional, Sequence, Callable
from torchio.transforms.transform import TypeMaskingMethod 
from torchio import Subject, Image

from med_slim.utils.preprocessing.slice_axis_resolver import resolve_slice_axis

TypeRangeFloat = Tuple[float, float]  # type: ignore
TypeTripletInt = Union[int, Tuple[int, int, int], Sequence[int]]  # type: ignore

def _get_affine(subject: tio.Subject) -> np.ndarray:
    affine = np.asarray(next(iter(subject.get_images_dict().values())).affine, dtype=float)
    if affine.shape != (4, 4) or not np.isfinite(affine[:3, :3]).all():
        return np.eye(4)
    return affine


def _slice_axis_from_subject(subject: tio.Subject, plane: str) -> int:
    """
    Return the through-plane axis for a canonicalized subject.
    """
    return resolve_slice_axis(
        np.asarray(subject.spatial_shape, dtype=float),
        np.asarray(subject.spacing, dtype=float),
        plane,
        affine=_get_affine(subject),
    )


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
        new_aff[:3, :3] = aff[:3, perm] # Reorder columns
        new_aff[:3, 3] = aff[:3, 3] # Keep translation
        image.affine = new_aff
    
    return subject


class EnsureSliceAxisLast(tio.Transform):
    """
    Ensure the slice (through-plane) axis is always at position 2 (last spatial dimension).
    Should be applied after ToCanonical() to ensure consistent axis ordering
    regardless of the original acquisition plane (axial, sagittal, coronal).
    
    After this transform:
    - Image shape is (C, W, H, D)
    - Spacing order matches the data order
    - The affine matrix is updated to maintain physical coordinates
    """
    def __init__(self, plane: str, **kwargs):
        super().__init__(**kwargs)
        self.plane = plane

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        slice_axis = _slice_axis_from_subject(subject, plane=self.plane)
        return _permute_slice_to_last(subject, slice_axis)

class CropEmptySlices(tio.Transform):
    """
    Remove near-empty slices from the edges of a volume along the slice axis.

    Applied after EnsureSliceAxisLast() so slices sit at position 2.
    Only trims contiguous runs of empty slices from the start and end;
    internal empty slices are preserved.

    Args:
        min_foreground_fraction: A slice is kept if at least this fraction
            of its voxels are above the background threshold (default 0.01 = 1%).
        min_slices_kept: Never trim below this many slices as a safety guard.
    """

    def __init__(
        self,
        min_foreground_fraction: float = 0.01,
        min_slices_kept: int = 10,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.min_foreground_fraction = min_foreground_fraction
        self.min_slices_kept = min_slices_kept

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        first_image = next(iter(subject.get_images_dict().values()))
        data = first_image.data  # (C, W, H, D)
        D = data.shape[-1]

        if D <= self.min_slices_kept:
            return subject

        bg_val = data.min().item()
        fg_fractions = (data[0] > bg_val).float().mean(dim=(0, 1))  # (D,)

        non_empty = fg_fractions >= self.min_foreground_fraction
        if not non_empty.any():
            return subject

        indices = non_empty.nonzero(as_tuple=True)[0]
        first = indices[0].item()
        last = indices[-1].item()

        keep = last - first + 1
        if keep < self.min_slices_kept:
            mid = (first + last) // 2
            first = max(0, mid - self.min_slices_kept // 2)
            last = min(D - 1, first + self.min_slices_kept - 1)

        if first == 0 and last == D - 1:
            return subject

        # tio.Crop((crop_ini_x, crop_fin_x, crop_ini_y, crop_fin_y, crop_ini_z, crop_fin_z))
        crop_end = D - 1 - last
        cropper = tio.Crop((0, 0, 0, 0, first, crop_end))
        return cropper(subject)


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


class ClipIntensity(tio.Transform):
    """
    Optionally clamp intensity values to a fixed range.

    Either bound may be left open by passing ``None``. Intended for
    model-specific medical preprocessing:

    - CT foundation models often expect Hounsfield units clamped to a physical
      range such as `[-1000, 1000]` before normalization.
    - Curia's `clip_below_air` option only clamps the lower bound to air
      (`min_value=-1000`) for CT. For MRI, set `enabled=False` so this transform acts as a no-op.
    """

    def __init__(
        self,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        enabled: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if enabled and min_value is None and max_value is None:
            raise ValueError("ClipIntensity requires at least one of min_value or max_value.")
        self.min_value = min_value
        self.max_value = max_value
        self.enabled = enabled
        self.args_names = ['min_value', 'max_value', 'enabled']

    def apply_transform(self, subject: Subject) -> Subject:
        if not self.enabled:
            return subject
        for image in subject.get_images_dict().values():
            data = image.data.float()
            data = torch.clamp(data, min=self.min_value, max=self.max_value)
            image.set_data(data)
        return subject


class PerSliceZScore(tio.Transform):
    """
    Z-score each in-plane slice independently, per channel.

    Operates on TorchIO data shaped ``(C, W, H, D)`` and normalizes over the
    in-plane axes ``(W, H)`` while keeping every slice along ``D`` independent.
    Apply after ``EnsureSliceAxisLast`` so the slice axis is last. Matches
    ``CuriaImageProcessor._zscore_per_image`` (unbiased std, blank slices left
    mean-subtracted only).
    """

    def __init__(self, eps: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.args_names = ['eps']

    def apply_transform(self, subject: Subject) -> Subject:
        for image in subject.get_images_dict().values():
            data = image.data.float()  # (C, W, H, D)
            mean = data.mean(dim=(1, 2), keepdim=True)
            std = data.std(dim=(1, 2), keepdim=True)
            std = torch.where(std < self.eps, torch.ones_like(std), std)
            image.set_data((data - mean) / std)
        return subject


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
        plane: str,
        padding_mode: Union[str, float] = 0,
        random_center: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_w, self.target_h, self.target_d = target_shape
        self.padding_mode = padding_mode
        self.random_center = random_center
        self.plane = plane

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        subject.check_consistent_space()
        
        slice_axis = _slice_axis_from_subject(subject, plane=self.plane)
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
        
    Args:
        target_shape_2d: Target shape for in-plane (W, H) dimensions
        padding_mode: Padding mode (see tio.Pad for options)
        random_center: If True, randomly center the crop/pad; if False, center crop/pad
        plane: Acquisition plane ('axial', 'sagittal', 'coronal')
    """

    def __init__(
        self,
        target_shape_2d: Union[int, Tuple[int, int], None],
        plane: str,
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
        self.plane = plane

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        subject.check_consistent_space()
        
        # Get current spatial shape
        current_shape = np.array(subject.spatial_shape)
        
        # Detect slice axis
        slice_axis = _slice_axis_from_subject(subject, plane=self.plane)
        
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
        plane: str,
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
        self.plane = plane
    
    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        current_shape = np.array(subject.spatial_shape)
        current_spacing = np.array(subject.spacing)
        slice_axis = _slice_axis_from_subject(subject, plane=self.plane)
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
        plane: str,
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
        self.plane = plane
    
    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        slice_axis = _slice_axis_from_subject(subject, plane=self.plane)
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
    Adaptive preprocessing that chooses optimal strategy based on source resolution and target FM requirements.
    
    Strategy:
    - Upsampling needed (target > source): CropOrPad (pad preserves native pixels, no interpolation)
    - Minor downsampling (0.7 < scale <= 1.0): CropOrPad (small crop, preserves native resolution)
    - Major downsampling (scale <= 0.7): CropOrPad to intermediate size, 
      then ResizeInPlane to exact target size
    """
    
    def __init__(
        self,
        target_size: Tuple[int, int],
        plane: str,
        num_slices: Optional[int] = None,
        padding_mode: str = 'minimum',
        resize_mode: str = 'bilinear',
        max_resize_ratio: float = 2,
        **kwargs
    ):
        """
        Args:
            target_size: Target (W, H) in-plane size for the FM
            plane: Acquisition plane ('axial', 'sagittal', 'coronal')
            num_slices: Target number of depth slices (None = keep original)
            padding_mode: Padding mode for CropOrPad ('minimum', 'constant', etc.)
            resize_mode: Interpolation mode for ResizeInPlane ('bilinear', 'bicubic', etc.)
            max_resize_ratio: Maximum allowed resize ratio for major downsampling.
                              The intermediate crop size = min(source, target * max_resize_ratio).
                              Lower values favor pixel quality, higher values favor FOV preservation.
            plane: Acquisition plane ('axial', 'sagittal', 'coronal')
        """
        super().__init__(**kwargs)
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        self.target_w, self.target_h = target_size
        self.num_slices = num_slices
        self.padding_mode = padding_mode
        self.resize_mode = resize_mode
        self.max_resize_ratio = max_resize_ratio
        self.plane = plane
    
    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        current_shape = np.array(subject.spatial_shape)
        
        slice_axis = _slice_axis_from_subject(subject, plane=self.plane)
        in_plane_axes = [i for i in range(3) if i != slice_axis]
        
        W_source = current_shape[in_plane_axes[0]]
        H_source = current_shape[in_plane_axes[1]]
        D_source = current_shape[slice_axis]
        
        # Calculate scale factors
        scale_w = self.target_w / W_source
        scale_h = self.target_h / H_source
        avg_scale = (scale_w + scale_h) / 2
        
        target_d = self.num_slices if self.num_slices else D_source
        
        if avg_scale > 0.7:
            target_shape = [0, 0, 0]
            target_shape[in_plane_axes[0]] = self.target_w
            target_shape[in_plane_axes[1]] = self.target_h
            target_shape[slice_axis] = target_d
            
            transform = tio.CropOrPad(
                target_shape=tuple(target_shape),
                padding_mode=self.padding_mode
            )
            subject = transform(subject)
        else:
            # Inspired by torchvision's RandomResizedCrop concept:
            # Step 1: CropOrPad to intermediate in-plane resolution and target depth.
            #         large sources get more crop, small sources get less.
            # Step 2: Resize from intermediate to exact target.
            intermediate_w = min(W_source, int(self.target_w * self.max_resize_ratio))
            intermediate_h = min(H_source, int(self.target_h * self.max_resize_ratio))
            
            intermediate_shape = [0, 0, 0]
            intermediate_shape[in_plane_axes[0]] = intermediate_w
            intermediate_shape[in_plane_axes[1]] = intermediate_h
            intermediate_shape[slice_axis] = target_d
            
            # Step 1: CropOrPad to intermediate size (handles both in-plane and depth)
            crop_or_pad = tio.CropOrPad(
                target_shape=tuple(intermediate_shape),
                padding_mode=self.padding_mode
            )
            subject = crop_or_pad(subject)
            
            # Step 2: Resize to exact FM target size (capped at max_resize_ratio)
            resize = ResizeInPlane(
                target_size=(self.target_h, self.target_w),
                mode=self.resize_mode,
                plane=self.plane,
            )
            subject = resize(subject)
        
        return subject