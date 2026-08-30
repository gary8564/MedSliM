"""Assemble per-slice Curia CLS/patch tokens into a volume sequence for the classifier."""

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch

CURIA_TOKEN_SOURCES = ("cls", "patch")
PATCH_POOLING_MODES = ("cls_only", "raw", "per_slice", "volume")


def validate_token_source(token_source: Sequence[str]) -> Tuple[str, ...]:
    """Return ``token_source`` as ``("cls",)``, ``("patch",)``, or ``("cls", "patch")``. List order is ignored."""
    if isinstance(token_source, str) or not isinstance(token_source, (list, tuple)):
        raise ValueError(
            f"token_source must be a list of {list(CURIA_TOKEN_SOURCES)} entries, "
            f"got {token_source!r} of type {type(token_source).__name__}."
        )
    entries = list(token_source)
    if not entries:
        raise ValueError(
            f"token_source must not be empty. Choose from {list(CURIA_TOKEN_SOURCES)}."
        )
    for entry in entries:
        if not isinstance(entry, str) or entry not in CURIA_TOKEN_SOURCES:
            raise ValueError(
                f"Unknown token_source entry {entry!r}. "
                f"Only {list(CURIA_TOKEN_SOURCES)} are allowed."
            )
    duplicates = sorted({e for e in entries if entries.count(e) > 1})
    if duplicates:
        raise ValueError(
            f"token_source must not repeat entries, got duplicates {duplicates} "
            f"in {entries}."
        )
    return tuple(source for source in CURIA_TOKEN_SOURCES if source in entries)


def resolve_patch_pooling(
    token_source: Sequence[str],
    use_avgpool_per_slice: bool = False,
    use_avgpool_on_the_volume: bool = False,
) -> str:
    """Choose patch pooling: per-slice mean, volume mean, or raw tokens. Ignored for CLS-only."""
    sources = validate_token_source(token_source)
    if "patch" not in sources:
        return "cls_only"
    if use_avgpool_per_slice:
        return "per_slice"
    if use_avgpool_on_the_volume:
        return "volume"
    return "raw"


def slice_positional_embeddings(
    num_slices: int,
    embed_dim: int,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sinusoidal embeddings over discrete slice positions ``0 .. num_slices - 1``."""
    positions = torch.arange(num_slices, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2, device=device, dtype=dtype)
        * (-math.log(10000.0) / embed_dim)
    )
    pe = torch.zeros(num_slices, embed_dim, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(positions * div_term)
    pe[:, 1::2] = torch.cos(positions * div_term[: pe[:, 1::2].shape[1]])
    return pe


def select_slice_indices(num_valid_slices: int, num_slices: Optional[int]) -> List[int]:
    """Centre-window slice indices, or all slices when ``num_slices`` is None."""
    if num_valid_slices <= 0:
        raise ValueError(f"num_valid_slices must be >= 1, got {num_valid_slices}.")
    if num_slices is None:
        return list(range(num_valid_slices))
    if num_slices <= 0:
        raise ValueError(f"num_slices must be >= 1 or None, got {num_slices}.")
    middle = num_valid_slices // 2
    start = middle - num_slices // 2
    return [i for i in range(start, start + num_slices) if 0 <= i < num_valid_slices]


def _add_slice_pe(features: torch.Tensor, enabled: bool) -> torch.Tensor:
    """Add one slice embedding per row of a ``[S, E]`` slice-level feature tensor."""
    if not enabled:
        return features
    num_slices, embed_dim = features.shape
    return features + slice_positional_embeddings(
        num_slices, embed_dim, device=features.device, dtype=features.dtype
    )


def assemble_volume_tokens(
    cls_tokens: Optional[torch.Tensor],
    patch_tokens: Optional[torch.Tensor],
    mode: str,
    add_slice_positional_embedding: bool = True,
) -> torch.Tensor:
    """Build a ``[T, E]`` sequence from one volume's selected CLS and/or patch tokens."""
    if mode not in PATCH_POOLING_MODES:
        raise ValueError(f"Unknown pooling mode {mode!r}. Choose from {list(PATCH_POOLING_MODES)}.")
    add_pe = add_slice_positional_embedding

    if mode == "cls_only":
        return _add_slice_pe(cls_tokens, add_pe)

    if mode == "per_slice":
        patch_features = _add_slice_pe(patch_tokens.mean(dim=1), add_pe)
        if cls_tokens is None:
            return patch_features
        cls_features = _add_slice_pe(cls_tokens, add_pe)
        return torch.stack((cls_features, patch_features), dim=1).flatten(0, 1)

    if mode == "volume":
        patch_features = patch_tokens.flatten(0, 1).mean(dim=0, keepdim=True)
        if cls_tokens is None:
            return patch_features
        return torch.cat((cls_tokens.mean(dim=0, keepdim=True), patch_features), dim=0)

    num_slices, patches_per_slice, embed_dim = patch_tokens.shape
    patch_features = patch_tokens
    if add_pe:
        pe = slice_positional_embeddings(
            num_slices, embed_dim, device=patch_tokens.device, dtype=patch_tokens.dtype
        )
        patch_features = patch_features + pe.unsqueeze(1)
    if cls_tokens is None:
        return patch_features.reshape(num_slices * patches_per_slice, embed_dim)
    cls_features = _add_slice_pe(cls_tokens, add_pe)
    return torch.cat((cls_features.unsqueeze(1), patch_features), dim=1).reshape(
        num_slices * (patches_per_slice + 1), embed_dim
    )


def pad_token_sequences(
    sequences: Sequence[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad ``[T_i, E]`` sequences to ``[B, T_max, E]``. Mask is True for real tokens."""
    max_tokens = max(seq.shape[0] for seq in sequences)
    embed_dim = sequences[0].shape[-1]
    reference = sequences[0]
    tokens = reference.new_zeros(len(sequences), max_tokens, embed_dim)
    mask = torch.zeros(len(sequences), max_tokens, dtype=torch.bool, device=reference.device)
    for i, seq in enumerate(sequences):
        tokens[i, : seq.shape[0]] = seq
        mask[i, : seq.shape[0]] = True
    return tokens, mask


def extract_curia_volume_tokens(
    cls_per_slice: Optional[torch.Tensor] = None,
    patch_per_slice: Optional[torch.Tensor] = None,
    slice_mask: Optional[torch.Tensor] = None,
    token_source: Sequence[str] = ("patch",),
    use_avgpool_per_slice: bool = False,
    use_avgpool_on_the_volume: bool = False,
    num_slices: Optional[int] = None,
    add_slice_positional_embedding: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select slices, pool patches, optionally prepend CLS, and pad to a batch."""
    sources = validate_token_source(token_source)
    mode = resolve_patch_pooling(sources, use_avgpool_per_slice, use_avgpool_on_the_volume)
    want_cls = "cls" in sources
    want_patch = "patch" in sources

    if want_cls and cls_per_slice is None:
        raise ValueError("token_source contains 'cls' but cls_per_slice is None.")
    if want_patch and patch_per_slice is None:
        raise ValueError("token_source contains 'patch' but patch_per_slice is None.")
    if want_cls and cls_per_slice.ndim != 3:
        raise ValueError(
            f"cls_per_slice must be [B, D, E], got shape {tuple(cls_per_slice.shape)}."
        )
    if want_patch and patch_per_slice.ndim != 4:
        raise ValueError(
            f"patch_per_slice must be [B, D, P, E], got shape {tuple(patch_per_slice.shape)}."
        )
    if want_cls and want_patch:
        if cls_per_slice.shape[:2] != patch_per_slice.shape[:2]:
            raise ValueError(
                f"cls_per_slice {tuple(cls_per_slice.shape)} and patch_per_slice "
                f"{tuple(patch_per_slice.shape)} disagree on batch/depth."
            )
        if cls_per_slice.shape[-1] != patch_per_slice.shape[-1]:
            raise ValueError(
                "CLS and patch tokens must share the embedding dim to be "
                f"sequence-concatenated, got {cls_per_slice.shape[-1]} and "
                f"{patch_per_slice.shape[-1]}."
            )

    reference = cls_per_slice if want_cls else patch_per_slice
    batch_size, depth = reference.shape[0], reference.shape[1]
    if slice_mask is not None:
        slice_mask = slice_mask.to(device=reference.device, dtype=torch.bool)
        if tuple(slice_mask.shape) != (batch_size, depth):
            raise ValueError(
                f"slice_mask must be [B, D] = {(batch_size, depth)}, "
                f"got {tuple(slice_mask.shape)}."
            )

    sequences = []
    for sample in range(batch_size):
        valid = int(slice_mask[sample].sum()) if slice_mask is not None else depth
        indices = select_slice_indices(valid, num_slices)
        sequences.append(
            assemble_volume_tokens(
                cls_tokens=cls_per_slice[sample, indices] if want_cls else None,
                patch_tokens=patch_per_slice[sample, indices] if want_patch else None,
                mode=mode,
                add_slice_positional_embedding=add_slice_positional_embedding,
            )
        )
    return pad_token_sequences(sequences)


def describe_token_layout(
    token_source: Sequence[str],
    use_avgpool_per_slice: bool = False,
    use_avgpool_on_the_volume: bool = False,
    num_slices: Optional[int] = None,
) -> Dict[str, object]:
    """Summarize the resolved token configuration for logging and cache metadata."""
    sources = validate_token_source(token_source)
    return {
        "token_source": list(sources),
        "pooling_mode": resolve_patch_pooling(
            sources, use_avgpool_per_slice, use_avgpool_on_the_volume
        ),
        "num_slices": "all" if num_slices is None else int(num_slices),
    }
