"""Curia token cache: NIfTI → image processor → frozen Curia backbone → safetensors."""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torchio as tio
from einops import rearrange
from safetensors import safe_open
from safetensors.torch import save_file
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from baselines.curia.utils import (
    resolve_patch_pooling,
    select_slice_indices,
    validate_token_source,
)
from med_slim.data.slice_dataset import SliceDataset
from med_slim.logging.setup import init_logging
from med_slim.utils.preprocessing.augmentation import CropEmptySlices, EnsureSliceAxisLast

init_logging()
logger = logging.getLogger(__name__)

# Pinned so the cache is reproducible even if the Hub repo moves forward.
CURIA_HUB_REPO = "raidium/curia"
CURIA_HUB_REVISION = "9657dc56276bc6c9503ef6f8d060879c8bee482f"

CLS_PER_SLICE = "cls_per_slice"
PATCH_MEAN_PER_SLICE = "patch_mean_per_slice"
CENTER_PATCH_TOKENS = "center_patch_tokens"
CENTER_SLICE_INDICES = "center_slice_indices"

DEFAULT_CENTER_NUM_SLICES = 3


def load_curia_image_processor(
    model_repo: str = CURIA_HUB_REPO,
    revision: str = CURIA_HUB_REVISION,
    local_cache_dir: Optional[str] = None,
):
    """Load the image processor published alongside the Curia weights."""
    logger.info(
        "Loading official Curia image processor from '%s' (revision %s) ...",
        model_repo,
        revision,
    )
    return AutoImageProcessor.from_pretrained(
        model_repo,
        revision=revision,
        trust_remote_code=True,
        cache_dir=local_cache_dir,
    )


def build_curia_orientation_transform(
    plane: str,
    crop_empty_slices: bool = False,
) -> tio.Compose:
    """Canonical orientation and slice-axis last."""
    transforms: List[tio.Transform] = [tio.ToCanonical(), EnsureSliceAxisLast(plane=plane)]
    if crop_empty_slices:
        transforms.append(CropEmptySlices())
    return tio.Compose(transforms)


def preprocess_curia_volume(
    volume: torch.Tensor,
    crop_size: int = 512,
    eps: float = 1e-6,
    clip_below_air: bool = False,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Resize every slice on `device` (bicubic 512) then whole-volume z-score.
    """
    if volume.ndim != 4 or volume.shape[0] != 1:
        raise ValueError(
            f"Expected a grayscale [1, W, H, D] volume, got shape {tuple(volume.shape)}."
        )
    slices = volume[0].permute(2, 1, 0).unsqueeze(1).contiguous().float()
    if device is not None:
        slices = slices.to(device, non_blocking=True)
    if clip_below_air:
        slices = slices.clamp_min(-1000.0)
    crop_size = int(crop_size)
    if slices.shape[-2] != crop_size or slices.shape[-1] != crop_size:
        slices = torch.nn.functional.interpolate(
            slices,
            size=(crop_size, crop_size),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
    mean = slices.mean()
    std = slices.std()
    if float(std) < eps:
        return slices - mean
    return (slices - mean) / std


def load_curia_backbone(
    model_repo: str = CURIA_HUB_REPO,
    revision: str = CURIA_HUB_REVISION,
    local_cache_dir: Optional[str] = None,
) -> torch.nn.Module:
    """Load the frozen Curia ViT published with the official processor."""
    logger.info(
        "Loading official Curia backbone from '%s' (revision %s) ...",
        model_repo,
        revision,
    )
    model = AutoModel.from_pretrained(
        model_repo,
        revision=revision,
        trust_remote_code=True,
        cache_dir=local_cache_dir,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _pool_patch_tokens(
    patch_tokens: torch.Tensor,
    spatial_pool_kernel_size: Optional[int],
) -> torch.Tensor:
    if spatial_pool_kernel_size is None or spatial_pool_kernel_size == 1:
        return patch_tokens
    num_patches = patch_tokens.shape[1]
    spatial_dim = int(num_patches ** 0.5)
    if spatial_dim * spatial_dim != num_patches:
        raise ValueError(f"Expected a square patch grid, got {num_patches} patch tokens.")
    patch_grid = rearrange(
        patch_tokens, "n (h w) e -> n e h w", h=spatial_dim, w=spatial_dim
    )
    pooled = torch.nn.functional.avg_pool2d(
        patch_grid, kernel_size=spatial_pool_kernel_size, stride=spatial_pool_kernel_size
    )
    return rearrange(pooled, "n e h w -> n (h w) e")


def encode_slice_batch(
    backbone: torch.nn.Module,
    pixel_values: torch.Tensor,
    spatial_pool_kernel_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode [N, 1, H, W] slices → CLS [N, E] and patches [N, P, E]."""
    if pixel_values.ndim != 4:
        raise ValueError(
            f"Expected slice batch [N, 1, H, W], got shape {tuple(pixel_values.shape)}."
        )
    if pixel_values.shape[1] != 1:
        raise ValueError(
            f"Expected grayscale slices (C=1), got C={pixel_values.shape[1]}."
        )
    outputs = backbone(pixel_values=pixel_values, return_dict=True)
    return (
        outputs.last_hidden_state[:, 0, :],
        _pool_patch_tokens(outputs.last_hidden_state[:, 1:, :], spatial_pool_kernel_size),
    )


@torch.no_grad()
def encode_curia_volume(
    backbone: torch.nn.Module,
    pixel_values: torch.Tensor,
    device: Optional[torch.device] = None,
    slice_batch_size: int = 64,
    amp_dtype: Optional[torch.dtype] = None,
    spatial_pool_kernel_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode preprocessed slice CLS tokens and patch tokens in depth chunks."""
    if slice_batch_size < 1:
        raise ValueError(f"slice_batch_size must be >= 1, got {slice_batch_size}.")
    device = device or next(backbone.parameters()).device
    cls_chunks, patch_chunks = [], []
    for start in range(0, pixel_values.shape[0], slice_batch_size):
        chunk = pixel_values[start : start + slice_batch_size]
        if chunk.device != device:
            chunk = chunk.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            cls_tokens, patch_tokens = encode_slice_batch(
                backbone, chunk, spatial_pool_kernel_size=spatial_pool_kernel_size
            )
        cls_chunks.append(cls_tokens)
        patch_chunks.append(patch_tokens)
    cls_tokens = torch.cat(cls_chunks, dim=0).float().cpu()
    patch_tokens = torch.cat(patch_chunks, dim=0).float().cpu()
    return cls_tokens, patch_tokens


def curia_cache_dir(
    cache_root: str,
    dataset_name: str,
    split: str,
    plane: str,
) -> Path:
    return Path(cache_root) / dataset_name / split / plane


def cache_manifest_path(cache_dir: str | Path) -> Path:
    return Path(cache_dir) / "manifest.json"


def cache_manifest(
    crop_size: int,
    center_num_slices: int,
    revision: str,
    crop_empty_slices: bool = False,
    spatial_pool_kernel_size: Optional[int] = None,
) -> Dict[str, str]:
    return {
        "crop_size": str(int(crop_size)),
        "center_num_slices": str(int(center_num_slices)),
        "hub_revision": str(revision),
        "crop_empty_slices": str(bool(crop_empty_slices)),
        "spatial_pool_kernel_size": str(spatial_pool_kernel_size),
    }


def write_cache_manifest(cache_dir: str | Path, manifest: Dict[str, str]) -> None:
    path = cache_manifest_path(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def assert_cache_manifest_compatible(cache_dir: str | Path, expected: Dict[str, str]) -> None:
    path = cache_manifest_path(cache_dir)
    if not path.is_file():
        return
    found = json.loads(path.read_text())
    mismatch = {
        key: (found.get(key), expected[key])
        for key in expected
        if str(found.get(key)) != str(expected[key])
    }
    if mismatch:
        raise ValueError(
            f"Existing Curia cache at {cache_dir} was built with different preprocessing "
            f"({mismatch}). Delete that folder to rebuild, or point --feature-cache-dir "
            "at a new root. Token recipes share this cache; do not nest it under an "
            "ablation tag."
        )


def write_curia_cache_entry(
    path: Path,
    cls_per_slice: torch.Tensor,
    patch_mean_per_slice: torch.Tensor,
    center_patch_tokens: torch.Tensor,
    center_slice_indices: Sequence[int],
    metadata: Dict[str, str],
) -> None:
    """Write one volume as safetensors plus string metadata."""
    tensors = {
        CLS_PER_SLICE: cls_per_slice.contiguous(),
        PATCH_MEAN_PER_SLICE: patch_mean_per_slice.contiguous(),
        CENTER_PATCH_TOKENS: center_patch_tokens.contiguous(),
        CENTER_SLICE_INDICES: torch.tensor(list(center_slice_indices), dtype=torch.int64),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(path), metadata={k: str(v) for k, v in metadata.items()})


def read_curia_cache_entry(
    path: str | Path,
    keys: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, str]]:
    """Read a cache entry; pass keys to skip unused tensors (e.g. raw patches)."""
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        available = set(handle.keys())
        wanted = list(available if keys is None else keys)
        missing = [key for key in wanted if key not in available]
        if missing:
            raise KeyError(
                f"Curia cache entry {path} is missing {missing}. "
                f"Available tensors: {sorted(available)}."
            )
        tensors = {key: handle.get_tensor(key) for key in wanted}
        return tensors, dict(handle.metadata() or {})


def ensure_curia_token_cache(
    data_dir: str,
    cache_root: str,
    dataset_name: str,
    split: str,
    plane: str,
    model_repo: str = CURIA_HUB_REPO,
    revision: str = CURIA_HUB_REVISION,
    local_cache_dir: Optional[str] = None,
    crop_empty_slices: bool = False,
    center_num_slices: int = DEFAULT_CENTER_NUM_SLICES,
    spatial_pool_kernel_size: Optional[int] = None,
    sample_ids: Optional[Sequence[str]] = None,
    device: Optional[torch.device] = None,
    slice_batch_size: int = 32,
    amp_dtype: Optional[torch.dtype] = None,
    mri_sequences: Optional[str | Sequence[str]] = None,
    overwrite: bool = False,
    crop_size: int = 512,
) -> Path:
    """Fill missing cache files for one split; skip the backbone if everything is present."""
    if center_num_slices < 1:
        raise ValueError(f"center_num_slices must be >= 1, got {center_num_slices}.")

    transform = build_curia_orientation_transform(plane, crop_empty_slices=crop_empty_slices)
    dataset = SliceDataset(
        path_root=data_dir,
        split=split,
        transform=transform,
        plane=plane,
        mri_sequences=mri_sequences,
    )
    wanted_ids = list(dataset.sample_ids if sample_ids is None else sample_ids)
    unknown = sorted(set(wanted_ids) - set(dataset.sample_ids))
    if unknown:
        raise ValueError(
            f"{len(unknown)} requested sample ids are absent from "
            f"{data_dir}/{split}/{plane} (first few: {unknown[:5]})."
        )

    out_dir = curia_cache_dir(cache_root, dataset_name, split, plane)
    out_dir.mkdir(parents=True, exist_ok=True)
    expected_manifest = cache_manifest(
        crop_size=crop_size,
        center_num_slices=center_num_slices,
        revision=revision,
        crop_empty_slices=crop_empty_slices,
        spatial_pool_kernel_size=spatial_pool_kernel_size,
    )
    assert_cache_manifest_compatible(out_dir, expected_manifest)

    pending = [
        uid for uid in wanted_ids
        if overwrite or not (out_dir / f"{uid}.safetensors").is_file()
    ]
    if not pending:
        write_cache_manifest(out_dir, expected_manifest)
        logger.info(
            "Curia token cache complete for split '%s' (%d volumes) at %s",
            split,
            len(wanted_ids),
            out_dir,
        )
        return out_dir

    logger.info(
        "Building Curia token cache for split '%s': %d/%d volumes missing -> %s",
        split,
        len(pending),
        len(wanted_ids),
        out_dir,
    )
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    logger.info(
        "Encoding Curia tokens on %s (slice_batch_size=%d, amp=%s, crop_size=%d)",
        device,
        slice_batch_size,
        amp_dtype,
        crop_size,
    )
    backbone = load_curia_backbone(
        model_repo=model_repo,
        revision=revision,
        local_cache_dir=local_cache_dir,
    ).to(device)

    index_by_uid = {uid: i for i, uid in enumerate(dataset.sample_ids)}
    for uid in tqdm(pending, desc=f"Caching Curia tokens ({split})"):
        sample = dataset[index_by_uid[uid]]
        volume = sample["source"].tensor
        pixel_values = preprocess_curia_volume(
            volume, crop_size=crop_size, device=device,
        )
        cls_tokens, patch_tokens = encode_curia_volume(
            backbone,
            pixel_values,
            device=device,
            slice_batch_size=slice_batch_size,
            amp_dtype=amp_dtype,
            spatial_pool_kernel_size=spatial_pool_kernel_size,
        )
        if not torch.isfinite(cls_tokens).all() or not torch.isfinite(patch_tokens).all():
            raise ValueError(f"Non-finite Curia features for volume '{uid}'.")

        depth = cls_tokens.shape[0]
        center_indices = select_slice_indices(depth, center_num_slices)
        write_curia_cache_entry(
            path=out_dir / f"{uid}.safetensors",
            cls_per_slice=cls_tokens,
            patch_mean_per_slice=patch_tokens.mean(dim=1),
            center_patch_tokens=patch_tokens[center_indices],
            center_slice_indices=center_indices,
            metadata={
                "uid": uid,
                "plane": plane,
                "split": split,
                "dataset_name": dataset_name,
                "model_repo": model_repo,
                "hub_revision": revision,
                "processor_class": "torch_bicubic",
                "crop_size": crop_size,
                "num_slices": depth,
                "center_num_slices": center_num_slices,
                "patches_per_slice": patch_tokens.shape[1],
                "embed_dim": patch_tokens.shape[-1],
                "spatial_pool_kernel_size": spatial_pool_kernel_size,
                "crop_empty_slices": crop_empty_slices,
            },
        )

    write_cache_manifest(out_dir, expected_manifest)
    logger.info("Curia token cache ready at %s", out_dir)
    return out_dir


class CuriaTokenCacheDataset(Dataset):
    """
    Serve cached tokens with the slice window already applied.
    Returns cls_per_slice [S,E] and/or patch_per_slice [S,P,E] (P=1 when pooled). Raw patches are only cached for the centre window.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        sample_ids: Optional[Sequence[str]] = None,
        token_source: Sequence[str] = ("patch",),
        use_avgpool_per_slice: bool = False,
        use_avgpool_on_the_volume: bool = False,
        num_slices: Optional[int] = None,
    ):
        super().__init__()
        self.cache_dir = Path(cache_dir)
        self.token_source = validate_token_source(token_source)
        self.pooling_mode = resolve_patch_pooling(
            self.token_source, use_avgpool_per_slice, use_avgpool_on_the_volume
        )
        self.num_slices = num_slices

        if self.pooling_mode == "raw" and num_slices is None:
            raise ValueError(
                "Raw patch tokens over the full stack are not cached: the sequence "
                "would be num_slices x 1024 tokens per volume. Set num_slices (3 for "
                "the released kneeMRI recipe), or enable use_avgpool_per_slice / "
                "use_avgpool_on_the_volume to aggregate patches first."
            )

        if sample_ids is None:
            sample_ids = sorted(p.stem for p in self.cache_dir.glob("*.safetensors"))
        self.sample_ids = list(sample_ids)
        if not self.sample_ids:
            raise ValueError(f"No cached Curia token files found in {self.cache_dir}.")

        self._keys = []
        if "cls" in self.token_source:
            self._keys.append(CLS_PER_SLICE)
        if "patch" in self.token_source:
            self._keys.append(
                CENTER_PATCH_TOKENS if self.pooling_mode == "raw" else PATCH_MEAN_PER_SLICE
            )
        if self.pooling_mode == "raw":
            self._keys.append(CENTER_SLICE_INDICES)
        # Volume depth drives slice selection; take it from any full-depth tensor
        # already being read, and fall back to metadata for raw patches only.
        self._depth_key = next(
            (key for key in (CLS_PER_SLICE, PATCH_MEAN_PER_SLICE) if key in self._keys), None
        )

    def __len__(self) -> int:
        return len(self.sample_ids)

    def cache_path(self, uid: str) -> Path:
        return self.cache_dir / f"{uid}.safetensors"

    def _slice_indices(self, depth: int) -> List[int]:
        return select_slice_indices(depth, self.num_slices)

    def load_tokens(self, uid: str) -> Dict[str, torch.Tensor]:
        """Load one volume, restricted to the configured slice window."""
        tensors, metadata = read_curia_cache_entry(self.cache_path(uid), keys=self._keys)
        if self._depth_key is not None:
            depth = tensors[self._depth_key].shape[0]
        elif "num_slices" in metadata:
            depth = int(metadata["num_slices"])
        else:
            raise ValueError(
                f"Cache entry {self.cache_path(uid)} has no 'num_slices' metadata, so the "
                "volume depth needed for centre-window selection cannot be recovered."
            )
        indices = self._slice_indices(depth)
        item: Dict[str, torch.Tensor] = {}

        if self.pooling_mode == "raw":
            cached_indices = tensors[CENTER_SLICE_INDICES].tolist()
            if cached_indices != indices:
                raise ValueError(
                    f"Cached centre window {cached_indices} for '{uid}' does not match "
                    f"num_slices={self.num_slices} (expected {indices}). Rebuild the "
                    f"cache with center_num_slices={self.num_slices}; the cache was "
                    f"built with center_num_slices={metadata.get('center_num_slices')}."
                )
            if "patch" in self.token_source:
                item["patch_per_slice"] = tensors[CENTER_PATCH_TOKENS].float()
        elif "patch" in self.token_source:
            item["patch_per_slice"] = tensors[PATCH_MEAN_PER_SLICE][indices].float().unsqueeze(1)

        if "cls" in self.token_source:
            item["cls_per_slice"] = tensors[CLS_PER_SLICE][indices].float()
        return item

    def __getitem__(self, index: int) -> Dict[str, object]:
        uid = self.sample_ids[index]
        return {"uid": uid, **self.load_tokens(uid)}


def curia_token_collate_fn(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Pad per-slice tokens over depth (trailing) and build slice_mask."""
    uids = [sample["uid"] for sample in batch]
    reference_key = "cls_per_slice" if "cls_per_slice" in batch[0] else "patch_per_slice"
    depths = [sample[reference_key].shape[0] for sample in batch]
    max_depth = max(depths)

    collated: Dict[str, object] = {"uid": uids}
    for key in ("cls_per_slice", "patch_per_slice"):
        if key not in batch[0]:
            continue
        reference = batch[0][key]
        padded = reference.new_zeros(len(batch), max_depth, *reference.shape[1:])
        for i, sample in enumerate(batch):
            tensor = sample[key]
            padded[i, : tensor.shape[0]] = tensor
        collated[key] = padded

    slice_mask = torch.zeros(len(batch), max_depth, dtype=torch.bool)
    for i, depth in enumerate(depths):
        slice_mask[i, :depth] = True
    collated["slice_mask"] = slice_mask

    if "label" in batch[0]:
        collated["labels"] = torch.stack([sample["label"] for sample in batch])
    return collated


def main():
    parser = argparse.ArgumentParser(
        description="Precompute Curia tokens with the image processor.",
    )
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Preprocessed NIfTI root ({data_dir}/{split}/{plane}/*.nii.gz).")
    parser.add_argument("--cache-root", type=str, required=True,
                        help="Root directory for the token cache.")
    parser.add_argument("--dataset-name", type=str, required=True,
                        help="Dataset name used in the cache path.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--plane", type=str, default="sagittal",
                        choices=["axial", "sagittal", "coronal"])
    parser.add_argument("--model-repo", type=str, default=CURIA_HUB_REPO)
    parser.add_argument("--revision", type=str, default=CURIA_HUB_REVISION,
                        help="Pinned Hub revision of the weights and processor.")
    parser.add_argument("--local-cache-dir", type=str, default=None)
    parser.add_argument("--crop-empty-slices", action="store_true",
                        help="Trim near-empty edge slices before encoding.")
    parser.add_argument("--center-num-slices", type=int, default=DEFAULT_CENTER_NUM_SLICES,
                        help="Centre window size for which raw patch tokens are cached.")
    parser.add_argument("--spatial-pool-kernel-size", type=int, default=None,
                        help="Optional avg-pool kernel over the 2D patch grid.")
    parser.add_argument("--slice-batch-size", type=int, default=64)
    parser.add_argument("--crop-size", type=int, default=512,
                        help="Resize target for pretrained Curia image size.")
    parser.add_argument("--amp", type=str, default=None, choices=["fp16", "bf16"],
                        help="Autocast dtype for the frozen backbone.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute entries that already exist.")
    args = parser.parse_args()

    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(args.amp)
    ensure_curia_token_cache(
        data_dir=args.data_dir,
        cache_root=args.cache_root,
        dataset_name=args.dataset_name,
        split=args.split,
        plane=args.plane,
        model_repo=args.model_repo,
        revision=args.revision,
        local_cache_dir=args.local_cache_dir,
        crop_empty_slices=args.crop_empty_slices,
        center_num_slices=args.center_num_slices,
        spatial_pool_kernel_size=args.spatial_pool_kernel_size,
        slice_batch_size=args.slice_batch_size,
        amp_dtype=amp_dtype,
        overwrite=args.overwrite,
        crop_size=args.crop_size,
    )


if __name__ == "__main__":
    if os.environ.get("HF_TOKEN") is None:
        logger.warning(
            "HF_TOKEN is not set. Curia weights are gated (RAIL-M); accept the "
            "license at https://huggingface.co/raidium/curia first."
        )
    main()
