"""
Adapted from https://github.com/mueller-franzes/MST/blob/main/mst/data/datasets/augmentations/augmentations_3d.py
Müller-Franzes, Gustav, Firas Khader, Robert Siepmann, Tianyu Han, Jakob Nikolas Kather, Sven Nebelung and Daniel Truhn. 
"Medical Slice Transformer: Improved Diagnosis and Explainability on 3D Medical Images with DINOv2."
ArXiv abs/2411.15802 (2024).
"""
import torchio as tio 
from typing import Iterable, Tuple, Union, List, Optional, Sequence, Dict, Callable
from numbers import Number
import nibabel as nib 
import numpy as np
from torchio.transforms.transform import TypeMaskingMethod 
from torchio import Subject, Image
import torch 

TypeRangeFloat = Tuple[float, float]  # type: ignore
TypeTripletInt = Union[int, Tuple[int, int, int], Sequence[int]]  # type: ignore

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
        cutoff = torch.quantile(image_data.masked_select(mask).float(), torch.tensor(self.percentiles)/100.0)
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


class CropOrPad2D(tio.Transform):
    """
    Crop or pad only the first two spatial dimensions (W and H), leaving the third dimension (D) unchanged.
    
    Args:
        target_shape_2d: Target shape for (W, H) dimensions
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
        self.target_w, self.target_h = target_shape_2d
        self.padding_mode = padding_mode
        self.random_center = random_center

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        subject.check_consistent_space()
        
        # Get current spatial shape (W, H, D)
        current_shape = subject.spatial_shape
        current_w, current_h, current_d = current_shape
        
        # Create target shape keeping D unchanged
        target_shape = (self.target_w, self.target_h, current_d)
        
        # Use CropOrPad with the computed target shape
        crop_or_pad = CropOrPad(
            target_shape=target_shape,
            padding_mode=self.padding_mode,
            random_center=self.random_center,
        )
        
        return crop_or_pad(subject)