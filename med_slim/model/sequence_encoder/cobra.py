"""
Adapted from: https://github.com/KatherLab/COBRA/blob/main/cobra/model/model.py
Lenz, Tim, Peter Neidlinger, Marta Ligero, Georg Wölflein, Marko van Treeck and Jakob Nikolas Kather.
Unsupervised Foundation Model-Agnostic Slide-Level Representation Learning.
2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR): 30807-30817, 2024.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from einops import rearrange
from typing import List, Optional, Tuple
from contextlib import contextmanager

from .mamba2 import Mamba2Enc
from .transformer import TransformerEncoderLayer, VarlenTransformerEncoder
from med_slim.model.attention_pooling import (
    BatchedABMIL,
    InterSliceAggregator,
    WithinSliceAggregator,
)
from med_slim.logging.setup import init_logging

init_logging()
logger = logging.getLogger(__name__)

POOLING_TARGETS = ("post_encoder", "post_embed", "raw")


def _resolve_pooling_target(
    mode: str,
    pooling_target: Optional[str],
    regional_tokens: int,
    slice_pooling: str = "abmil",
) -> Optional[str]:
    if mode not in ("train", "inference"):
        raise ValueError(f"Invalid mode '{mode}'. Must be 'train' or 'inference'.")
    if slice_pooling not in ("abmil", "cross_attention", "cls"):
        raise ValueError(
            f"Invalid slice_pooling '{slice_pooling}'. Must be 'abmil', "
            "'cross_attention', or 'cls'."
        )
    if pooling_target is not None and pooling_target not in POOLING_TARGETS:
        raise ValueError(
            f"Invalid pooling_target '{pooling_target}'. Must be one of "
            f"{POOLING_TARGETS} or None."
        )
    if mode == "train":
        return None

    if slice_pooling == "cls":
        if pooling_target in ("post_embed", "raw"):
            warnings.warn(
                "slice_pooling='cls' always pools the sequence-encoder CLS token; "
                "pooling_target is not applicable."
            )
        return None

    resolved = pooling_target
    if resolved is None:
        resolved = (
            "raw" if slice_pooling == "abmil" and regional_tokens == 0 else "post_embed"
        )
    if regional_tokens > 0 and resolved == "raw":
        raise ValueError(
            "pooling_target='raw' is not supported with tiled multi-crop CLS "
            "(regional_tokens > 0). Region tokens are collapsed to embed_dim by the "
            "within-slice aggregator, so use pooling_target='post_encoder' or 'post_embed'."
        )
    if slice_pooling == "cross_attention" and resolved == "raw":
        raise ValueError(
            "pooling_target='raw' is only supported with slice_pooling='abmil'. "
            f"Got slice_pooling='{slice_pooling}'."
        )
    return resolved


def sinusoidal_position_encoding(
    positions: torch.Tensor,
    dim: int,
    base: float = 10000.0,
) -> torch.Tensor:
    """
    Compute sinusoidal positional encoding from continuous positions (e.g. mm).

    PE(p, 2i)   = sin(p / base^{2i/d})
    PE(p, 2i+1) = cos(p / base^{2i/d})

    Args:
        positions: Arbitrary-shape tensor of positions (e.g. in millimeters).
        dim: Embedding dimensionality (must be even).
        base: Frequency base (default 10000, following Vaswani et al.).

    Returns:
        Sinusoidal encoding with shape ``(*positions.shape, dim)``.
    """
    assert dim % 2 == 0, "Embedding dimension must be even for sinusoidal PE"
    half = dim // 2
    freq_indices = torch.arange(half, device=positions.device, dtype=positions.dtype)
    inv_freq = 1.0 / (base ** (freq_indices / half))  
    angles = positions.unsqueeze(-1) * inv_freq 
    return torch.cat([angles.sin(), angles.cos()], dim=-1) 


class Embed(nn.Module):
    def __init__(self, dim, embed_dim=1024, dropout=0.25):
        super(Embed, self).__init__()

        self.head = nn.Sequential(
             nn.LayerNorm(dim),
             nn.Linear(dim, embed_dim),
             nn.Dropout(dropout) if dropout else nn.Identity(),
             nn.SiLU(),
             nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x):
        return self.head(x)


class Cobra(nn.Module):
    """
    Cobra model for processing and aggregating embeddings with attention.
    
    This model utilizes separate embedding layers for different input dimensions, followed by a
    normalization layer and either a Mamba2 or Transformer encoder. 
    It then applies two options to aggregate the features:
    - BatchedABMIL: multi-head attention pooling, returns MIL attention weights [B, 1, num_slices]
    - cls token: learnable cls token pooling, returns CLS → slice attention from last transformer layer
    
    Example:
        >>> model = Cobra(sequence_encoder="transformer", slice_pooling="cls")
        >>> x = torch.randn(2, 100, 768)  # [batch, num_slices, embed_dim]
        >>> seq_lens = torch.tensor([80, 100])  # actual lengths before padding
        >>> # Get features
        >>> features = model(x, seq_lengths=seq_lens)
        >>> # Get attention map
        >>> attn = model(x, seq_lengths=seq_lens, get_attention=True)  # [B, 1, num_slices]
    """

    def __init__(
        self,
        embed_dim: int = 768,
        contrast_dim: int = 256,
        input_dims: List[int] = None,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.25,
        mode: str = "train",
        sequence_encoder: str = "mamba2",
        fm_pooling: str = "avg_pool",
        slice_pooling: str = "abmil",
        pooling_target: Optional[str] = None,
        raw_output_dim: int = None,
        physical_pe: bool = False,
        regional_tokens: int = 0,
        region_embedding: bool = False,
        **kwargs
    ):
        """
        Args:
            embed_dim: Dimensionality of the embedding vectors.
            contrast_dim: Dimensionality of the contrastive features.
            input_dims: List of input feature dimensions for embedding lookup.
            num_heads: Number of attention heads for ABMIL pooling.
            num_layers: Number of encoder layers.
            dropout: Dropout rate.
            mode: 'train' or 'inference'.
            sequence_encoder: 'mamba2' or 'transformer'.
            fm_pooling: 'avg_pool' or 'attention' in inference mode.
                - 'avg_pool': Average pool embeddings across foundation models.
                - 'attention': Learn attention weights to pool FM embeddings per slice (requires fine-tuning).
            slice_pooling: 'abmil', 'cross_attention', or 'cls' (cls requires transformer).
            pooling_target: Which representation level to aggregate at inference.
                Ignored in training mode, where Cobra always pools post-encoder hidden
                states for contrastive pretraining. When None, inference defaults are
                resolved by ``_resolve_pooling_target`` (see that helper for per-path
                defaults). Meaning depends on ``slice_pooling``:
                - 'abmil': attention weights come from encoder output; pooling_target
                  selects which features they aggregate ('raw', 'post_embed', or
                  'post_encoder'). 'raw' requires global-only features (regional_tokens=0).
                - 'cross_attention': attention is computed from encoder output; pooling_target
                  selects value features ('post_encoder' = encoder states, 'post_embed' =
                  pre-encoder slice features after within-slice aggregation for tiled inputs).
                - 'cls': not applicable, volume embedding is always the
                  sequence-encoder CLS token. Resolved pooling_target is None
            raw_output_dim: FM embedding dimension for the 'raw' pooling target.
                Required in inference mode when the resolved pooling_target is 'raw'.
            physical_pe: If True, add sinusoidal positional encoding based on physical slice positions (in mm) before the sequence encoder.
                Requires ``physical_positions`` to be passed during forward.
            regional_tokens: Number of regional crop tokens per slice for tiled multi-crop CLS (e.g. 4 for a 2x2 grid). 0 (default) disables tiled
                aggregation and keeps the global-only CLS pathway. When > 0, a WithinSliceAggregator collapses [B, num_slices, 1+regional_tokens, embed_dim] 
                region tokens into [B, num_slices, embed_dim] before the sequence encoder.
            region_embedding: If True (and regional_tokens > 0), add a small learned region embedding over the (global + regional) tokens so the aggregator can use quadrant identity.
            **kwargs: Additional encoder-specific parameters:
                - d_state: Mamba2 state dimension (default: 128)
                - dim_feedforward: Transformer FFN dimension (default: 4*embed_dim)
                - norm_first: Pre-LN transformer (default: True)
                - rotary_positional_encoding: RoPE config (default: None)
                - att_dim: ABMIL hidden dimension for slice and FM attention pooling (default: 256)
        """
        super().__init__()

        if input_dims is None:
            input_dims = [512, 768, 1024, 1152, 1376, 1536]

        assert mode in ["train", "inference"]
        assert sequence_encoder in ["mamba2", "transformer"]
        if mode == "inference":
            assert fm_pooling in ["avg_pool", "attention"], f"Invalid fm_pooling '{fm_pooling}'. Must be one of 'avg_pool', 'attention'."
        assert slice_pooling in ["abmil", "cross_attention", "cls"], (
            f"Invalid slice_pooling '{slice_pooling}'. Must be one of 'abmil', 'cross_attention', 'cls'."
        )
        if slice_pooling == "cls" and sequence_encoder != "transformer":
                raise ValueError(f"slice_pooling='cls' requires sequence_encoder='transformer'. Got {sequence_encoder}.")
        resolved_pooling_target = _resolve_pooling_target(
            mode,
            pooling_target,
            regional_tokens,
            slice_pooling=slice_pooling,
        )
        if (
            mode == "inference"
            and resolved_pooling_target == "raw"
            and slice_pooling not in ("cls", "cross_attention")
            and raw_output_dim is None
        ):
            raise ValueError("raw_output_dim is required when pooling_target='raw' in inference mode.")

        self.mode = mode
        self.embed_dim = embed_dim
        self.pooling_target = resolved_pooling_target
        self._raw_output_dim = raw_output_dim
        self.sequence_encoder = sequence_encoder
        self.fm_pooling = fm_pooling
        self.slice_pooling = slice_pooling
        self.physical_pe = physical_pe

        # Tiled multi-crop CLS: within-slice region aggregation
        self.regional_tokens = regional_tokens
        self.region_embedding = region_embedding
        self.within_slice_agg = None
        # Cached global-to-region attention with shape [B, num_slices, 1, num_tiled_regions-1] in padded mode
        # or [1, total_slices, 1, num_tiled_regions-1] in packed mode.
        self.region_attention = None
        if regional_tokens > 0:
            num_regions = 1 + regional_tokens  # global + regional crops
            self.within_slice_agg = WithinSliceAggregator(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                num_regions=num_regions if region_embedding else None,
            )

        self.embed = nn.ModuleDict({str(d): Embed(d, embed_dim) for d in input_dims})
        self.norm = nn.LayerNorm(embed_dim)

        if self.sequence_encoder == "mamba2":
            # Sequence encoder
            self.seq_enc = Mamba2Enc(
                embed_dim,
                embed_dim,
                n_classes=embed_dim,
                layer=num_layers,
                dropout=dropout,
                d_state=kwargs.get('d_state', 128),
            )
        else:
            # Standard TransformerEncoder for zero-padding
            enc_layer = TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=kwargs.get('dim_feedforward', 4 * embed_dim),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=kwargs.get('norm_first', True),
                rotary_positional_encoding=kwargs.get('rotary_positional_encoding', None),
            )
            self.seq_enc = nn.TransformerEncoder(
                enc_layer,
                num_layers=num_layers,
                enable_nested_tensor=False,
            )
            # VarlenTransformerEncoder for packed sequences using FlashAttention
            self.varlen_seq_enc = VarlenTransformerEncoder(
                d_model=embed_dim,
                nhead=num_heads,
                num_layers=num_layers,
                dim_feedforward=kwargs.get('dim_feedforward', 4 * embed_dim),
                dropout=dropout,
                activation="gelu",
                norm_first=kwargs.get('norm_first', True),
                rotary_positional_encoding=kwargs.get('rotary_positional_encoding', None),
            )
        # Optional CLS token
        self.cls_token = None
        if self.slice_pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        # Projection head for SSL pretraining
        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(4 * embed_dim, contrast_dim),
            nn.BatchNorm1d(contrast_dim),
        )

        # Slice-level pooling modules
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.attn = None
        self.cross_attn_pool = None
        if self.slice_pooling == "abmil":
            self.attn = nn.ModuleList([
                BatchedABMIL(
                    input_dim=self.head_dim,
                    hidden_dim=kwargs.get('att_dim', 256),
                    dropout=dropout,
                    n_heads=1,
                    activation='softmax',
                )
                for _ in range(self.num_heads)
            ])
        elif self.slice_pooling == "cross_attention":
            self.cross_attn_pool = InterSliceAggregator(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
        
        # FM attention pooling for inference mode
        # Learns to weight different foundation model embeddings at each slice position
        # fm_attn is randomly initialized in inference mode and not loaded from checkpoint.
        # This is intended for FINE-TUNING COBRA (where fm_attn will be trained).
        # For LINEAR PROBING (frozen COBRA), use fm_pooling='avg_pool' instead.
        self.fm_attn = None
        if mode == "inference" and fm_pooling == "attention":
            self.fm_attn = BatchedABMIL(
                input_dim=embed_dim,
                hidden_dim=kwargs.get('att_dim', 256),
                dropout=0.5,
                n_heads=1,
                activation='softmax',
            )

    def _apply_physical_pe(
        self,
        logits: torch.Tensor,
        physical_positions: torch.Tensor,
        seq_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Add sinusoidal PE keyed on physical positions (mm) to embedded features.

        PE at padded positions is zeroed out so that padding remains inert.
        Without this, PE(0) = [0, 1, 0, 1, ...] (cos(0)=1) would inject a
        non-zero signal into padded positions and make position 0mm ambiguous
        with padding.
        """
        pe = sinusoidal_position_encoding(physical_positions, self.embed_dim)
        if seq_lengths is not None and logits.dim() == 3:
            # Padded mode: zero out PE beyond each sample's real sequence length
            mask = self._build_mask(seq_lengths, logits.shape[1])  # [B, T]
            pe = pe * mask.unsqueeze(-1)  # [B, T, 1]
        return logits + pe

    @property
    def output_dim(self) -> int:
        """Dimension of the volume-level output embedding, accounting for pooling_target."""
        if self.slice_pooling == "abmil" and self.pooling_target == "raw":
            return self._raw_output_dim
        return self.embed_dim

    def _build_mask(self, seq_lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """Build boolean mask from sequence lengths. True = valid, False = padded."""
        batch_size = seq_lengths.size(0)
        # create position indices [0, 1, 2, ..., max_len-1] with shape [batch_size, max_len]
        positions = torch.arange(max_len, device=seq_lengths.device).unsqueeze(0).expand(batch_size, -1)
        # mask[b, t] = True if t < seq_lengths[b], else False
        mask = positions < seq_lengths.unsqueeze(1)
        return mask

    @contextmanager
    def _capture_attention(self, layer: nn.Module):
        """
        Context manager to capture attention weights from a transformer layer's self-attention.
        
        Handles:
        - Disabling PyTorch MHA fastpath for reliable weight extraction
        - Wrapping forward to return attention weights
        - Cleanup on exit
        
        Yields:
            List of attention weights after forward pass.
        """
        attn_weights = []
        self_attn = layer.self_attn
        
        # Disable fastpath (required for attention weight extraction)
        fastpath_enabled = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)
        
        # Wrap forward to return attention weights
        orig_forward = self_attn.forward
        def wrapped_forward(*args, **kwargs):
            kwargs['need_weights'] = True
            kwargs['average_attn_weights'] = False
            return orig_forward(*args, **kwargs)
        
        # Hook to capture attention weights
        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) == 2 and output[1] is not None:
                attn_weights.append(output[1].detach())
        
        self_attn.forward = wrapped_forward
        handle = self_attn.register_forward_hook(hook)
        
        try:
            yield attn_weights
        finally:
            self_attn.forward = orig_forward
            handle.remove()
            torch.backends.mha.set_fastpath_enabled(fastpath_enabled)

    def _extract_cls_attention(
        self, 
        logits: torch.Tensor, 
        mask: torch.Tensor | None,
        src_key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Extract cls → slice attention from last transformer layer.
        
        Args:
            logits: Input with CLS prepended [B, 1+num_slices, embed_dim]
            mask: Valid slice mask [B, num_slices] (True = valid)
            src_key_padding_mask: Transformer padding mask [B, 1+num_slices] (True = padded)
        
        Returns:
            Attention weights [B, 1, num_slices].
        """
        last_layer = self.seq_enc.layers[-1]
        
        with self._capture_attention(last_layer) as attn_weights:
            _ = self.seq_enc(logits, src_key_padding_mask=src_key_padding_mask)
        
        if not attn_weights:
            raise RuntimeError("Failed to capture attention weights from transformer.")
        
        # Extract cls to slices attention
        last_attn = attn_weights[-1]  # [B, num_heads, 1+T, 1+T]
        cls_to_slices = last_attn[:, :, 0, 1:]  # [B, num_heads, num_slices]
        cls_to_slices = cls_to_slices.mean(dim=1)  # Average over heads [B, num_slices]
        
        # Mask padded positions and normalize
        if mask is not None:
            cls_to_slices = cls_to_slices.masked_fill(~mask, 0.0)
        cls_to_slices = cls_to_slices / cls_to_slices.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        
        return cls_to_slices.unsqueeze(1)  # [B, 1, num_slices]

    def _embed_ssl_forward(self, x, input_feature_dims=None) -> torch.Tensor:
        """
        Foundation model feature embedding in SSL pretraining mode.

        Supports global-only CLS inputs:
        [B, num_slices, input_embed_dim] -> [B, num_slices, embed_dim]
        as well as tiled multi-crop CLS inputs:
        [B, num_slices, num_tiled_regions, input_embed_dim] -> [B, num_slices, num_tiled_regions, embed_dim]. 
        """
        if input_feature_dims is not None:
            assert len(x)==len(input_feature_dims), "Batch size mismatch between input x and input_feature_dims"
            logits = torch.concat([self.embed[str(input_feature_dims[i].item())](x[i][..., :input_feature_dims[i].item()]).unsqueeze(0) for i in range(len(x))], dim=0) # [B, num_slices, (num_tiled_regions,) embed_dim]
        else:
            logits = self.embed[str(x.shape[-1])](x)  # [B, num_slices, (num_tiled_regions,) embed_dim]
        return logits

    def _embed_ssl_forward_packed_global_only(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        input_feature_dims: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed packed global-only CLS features [total_slices, input_embed_dim]."""
        if input_feature_dims is None:
            return self.embed[str(x.shape[-1])](x)  # [total_slices, embed_dim]

        input_feature_dims = input_feature_dims.to(device=x.device, dtype=torch.long)
        total_seq_len = x.shape[0]
        seq_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(device=x.device)  # [B]
        unique_dims = input_feature_dims.unique()

        if len(unique_dims) == 1:
            feat_dim = unique_dims[0].item()
            return self.embed[str(feat_dim)](x[:, :feat_dim])

        slice_feat_dims = torch.repeat_interleave(input_feature_dims, seq_lens)  # [total_slices]
        logits = None
        for dim in unique_dims:
            feat_dim = dim.item()
            mask = slice_feat_dims == dim  # [total_slices]
            x_group = x[mask, :feat_dim]  # [num_slices_for_dim, input_embed_dim]
            embedded = self.embed[str(feat_dim)](x_group)  # [num_slices_for_dim, embed_dim]
            if logits is None:
                logits = torch.zeros(total_seq_len, self.embed_dim, device=x.device, dtype=embedded.dtype)
            logits[mask] = embedded
        return logits

    def _embed_ssl_forward_packed_tiled(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        input_feature_dims: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed packed tiled CLS features [total_slices, num_tiled_regions, input_embed_dim]."""
        if input_feature_dims is None:
            return self.embed[str(x.shape[-1])](x)  # [total_slices, num_tiled_regions, embed_dim]

        input_feature_dims = input_feature_dims.to(device=x.device, dtype=torch.long)
        total_seq_len, num_tiled_regions = x.shape[:2]
        seq_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(device=x.device)  # [B]
        unique_dims = input_feature_dims.unique()

        if len(unique_dims) == 1:
            feat_dim = unique_dims[0].item()
            return self.embed[str(feat_dim)](x[..., :feat_dim])

        slice_feat_dims = torch.repeat_interleave(input_feature_dims, seq_lens)  # [total_slices]
        logits = None
        for dim in unique_dims:
            feat_dim = dim.item()
            mask = slice_feat_dims == dim  # [total_slices]
            x_group = x[mask, :, :feat_dim]  # [num_slices_for_dim, num_tiled_regions, input_embed_dim]
            embedded = self.embed[str(feat_dim)](x_group)  # [num_slices_for_dim, num_tiled_regions, embed_dim]
            if logits is None:
                logits = torch.zeros(
                    total_seq_len,
                    num_tiled_regions,
                    self.embed_dim,
                    device=x.device,
                    dtype=embedded.dtype,
                )
            logits[mask] = embedded
        return logits
    
    def _embed_ssl_forward_packed(
        self, 
        x: torch.Tensor, 
        cu_seqlens: torch.Tensor,
        input_feature_dims: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Foundation model feature embedding in SSL mode (packed sequences).
        
        Args:
            x: Packed global-only tensor [total_slices, input_embed_dim] or
               packed tiled tensor [total_slices, num_tiled_regions, input_embed_dim].
            cu_seqlens: Cumulative sequence lengths [batch_size + 1], int32
            input_feature_dims: Feature dimensions per sample [batch_size] (optional)
        
        Returns:
            Embedded global-only features [total_slices, embed_dim] or tiled
            features [total_slices, num_tiled_regions, embed_dim].
        """
        if x.dim() == 2:
            return self._embed_ssl_forward_packed_global_only(x, cu_seqlens, input_feature_dims)
        if x.dim() == 3:
            return self._embed_ssl_forward_packed_tiled(x, cu_seqlens, input_feature_dims)
        raise ValueError(
            "Packed SSL input must be [total_slices, input_embed_dim] for global-only CLS "
            "or [total_slices, num_tiled_regions, input_embed_dim] for tiled multi-crop CLS."
        )

    def _collapse_tiled_regions(self, logits: torch.Tensor, *, packed: bool = False) -> torch.Tensor:
        """
        Collapse tiled per-slice tokens to one representation per slice.

        Padded mode expects [B, num_slices, num_tiled_regions, embed_dim].
        Packed mode expects [total_slices, num_tiled_regions, embed_dim].
        Global-only tensors are passed through unchanged when the model is not tiled.
        """
        tiled_rank = 3 if packed else 4
        global_rank = 2 if packed else 3

        if logits.dim() == tiled_rank:
            if self.within_slice_agg is None:
                shape_hint = (
                    "[total_slices, num_tiled_regions, embed_dim]"
                    if packed
                    else "[B, num_slices, num_tiled_regions, embed_dim]"
                )
                raise ValueError(
                    f"Received tiled features {shape_hint} but the model was built without "
                    "a within-slice aggregator. Set regional_tokens > 0 when constructing Cobra."
                )

            num_tiled_regions = logits.shape[1] if packed else logits.shape[2]
            expected_regions = 1 + self.regional_tokens
            if num_tiled_regions != expected_regions:
                raise ValueError(
                    f"Expected {expected_regions} tiled regions per slice "
                    f"(1 global + {self.regional_tokens} regional), got {num_tiled_regions}."
                )

            if packed:
                collapsed, self.region_attention = self.within_slice_agg(logits.unsqueeze(0))
                return collapsed.squeeze(0)

            logits, self.region_attention = self.within_slice_agg(logits)
            return logits

        if self.within_slice_agg is not None and logits.dim() == global_rank:
            raise ValueError(
                "regional_tokens > 0 but received global-only features. Provide tiled "
                "multi-crop CLS caches with shape "
                "[total_slices, num_tiled_regions, input_embed_dim] in packed mode or "
                "[B, num_slices, num_tiled_regions, input_embed_dim] in padded mode."
            )

        return logits

    def _pool_fm_embeddings(
        self,
        fm_embs: torch.Tensor,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Pool post-Embed representations across foundation models at each slice.

        Args:
            fm_embs: [num_fms, B, num_slices, embed_dim].
            return_weights: If True, also return FM weights [B, num_slices, num_fms].
        """
        num_fms, batch_size, num_slices, embed_dim = fm_embs.shape

        if num_fms == 1:
            pooled = fm_embs[0]
            weights = torch.ones(
                batch_size,
                num_slices,
                1,
                device=fm_embs.device,
                dtype=fm_embs.dtype,
            )
        elif self.fm_pooling == "avg_pool":
            pooled = fm_embs.mean(dim=0)
            weights = torch.full(
                (batch_size, num_slices, num_fms),
                1.0 / num_fms,
                device=fm_embs.device,
                dtype=fm_embs.dtype,
            )
        else:
            # Attention-weighted pooling across foundation models for each slice.
            fm_tokens = rearrange(fm_embs, 'k b t e -> (b t) k e')
            weights_flat = self.fm_attn(fm_tokens)  # [B*num_slices, num_fms, 1]
            pooled = torch.bmm(
                weights_flat.transpose(2, 1),
                fm_tokens,
            ).squeeze(1)
            pooled = rearrange(pooled, '(b t) e -> b t e', b=batch_size, t=num_slices)
            weights = rearrange(
                weights_flat.squeeze(-1),
                '(b t) k -> b t k',
                b=batch_size,
                t=num_slices,
            )

        if return_weights:
            return pooled, weights
        return pooled
        
    def _packed_to_padded(
        self, 
        packed: torch.Tensor, 
        cu_seqlens: torch.Tensor, 
        max_seqlen: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert packed tensor to padded tensor with mask.
        
        Args:
            packed: Packed tensor [total_slices, embed_dim]
            cu_seqlens: Cumulative sequence lengths [batch_size + 1]
            max_seqlen: Maximum sequence length
        
        Returns:
            padded: Padded tensor [batch_size, max_seqlen, embed_dim]
            mask: Boolean mask [batch_size, max_seqlen] (True = valid)
        """
        batch_size = cu_seqlens.shape[0] - 1
        dim = packed.shape[-1]
        total_len = packed.shape[0]
        
        # Calculate sequence lengths
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]  # [batch_size]
        
        # Vectorized mask creation
        mask = torch.arange(max_seqlen, device=packed.device).unsqueeze(0) < lengths.unsqueeze(1)
        
        # Create batch indices
        batch_indices = torch.repeat_interleave(
            torch.arange(batch_size, device=packed.device), 
            lengths
        )
        
        # Create position indices
        position_indices = torch.arange(total_len, device=packed.device) - cu_seqlens[:-1].repeat_interleave(lengths)
        
        # Scatter into padded tensor
        padded = torch.zeros(batch_size, max_seqlen, dim, device=packed.device, dtype=packed.dtype)
        padded[batch_indices, position_indices] = packed
        
        return padded, mask

    def _zero_padded_slices(
        self,
        logits: torch.Tensor,
        seq_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        """Zero-padded slice positions before within-slice aggregation or the sequence encoder."""
        if seq_lengths is None:
            return logits
        max_len = logits.shape[1]
        mask = self._build_mask(seq_lengths, max_len)
        if logits.dim() == 3:
            return logits * mask.unsqueeze(-1)
        if logits.dim() == 4:
            return logits * mask.unsqueeze(-1).unsqueeze(-1)
        return logits
            
    def _embed_inference_forward(
        self,
        x,
        seq_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Foundation model feature embedding in inference mode (zero padding).

        Each FM input is [B, num_slices, input_embed_dim] (global-only CLS) or
        [B, num_slices, num_tiled_regions, input_embed_dim] (tiled multi-crop CLS).
        """
        # Inference mode with ensembling feature embeddings from multiple slice encoders
        embedded_features = [self.embed[str(xi.shape[-1])](xi) for xi in x]  # List of K [B, num_slices, (num_tiled_regions,) embed_dim]
        fm_embs = torch.stack(embedded_features, dim=0)  # [K, B, num_slices, (num_tiled_regions,) embed_dim]
        assert fm_embs.shape[-1] == self.embed_dim, f"Expected embed_dim {self.embed_dim}, got {fm_embs.shape[-1]}"
        assert fm_embs.dim() in (4, 5), f"Expected 4 or 5 dimensions, got {fm_embs.dim()}"
        assert fm_embs.shape[0]==len(x), f"Expected length of input x {len(x)}, got {fm_embs.shape[0]}"

        if fm_embs.dim() == 5:
            if self.within_slice_agg is None:
                raise ValueError(
                    "Received tiled inference features "
                    "[K, B, num_slices, num_tiled_regions, embed_dim] but the model "
                    "was built without a within-slice aggregator. Set regional_tokens > 0."
                )

            num_fms, batch_size, num_slices, num_tiled_regions, embed_dim = fm_embs.shape
            expected_regions = 1 + self.regional_tokens
            if num_tiled_regions != expected_regions:
                raise ValueError(
                    f"Expected {expected_regions} tiled regions per slice "
                    f"(1 global + {self.regional_tokens} regional), got {num_tiled_regions}."
                )

            if seq_lengths is not None:
                mask = self._build_mask(seq_lengths, num_slices)
                fm_embs = fm_embs * mask.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

            # Collapse regions independently for each FM, then pool the resulting
            # per-FM slice representations. This keeps FM attention defined on
            # comparable [B, num_slices, embed_dim] slice embeddings.
            fm_slice_embs, per_fm_region_attention = self.within_slice_agg(
                fm_embs.reshape(num_fms * batch_size, num_slices, num_tiled_regions, embed_dim)
            )
            fm_slice_embs = fm_slice_embs.reshape(num_fms, batch_size, num_slices, embed_dim)
            per_fm_region_attention = per_fm_region_attention.reshape(
                num_fms,
                batch_size,
                num_slices,
                1,
                num_tiled_regions - 1,
            )

            logits, fm_weights = self._pool_fm_embeddings(fm_slice_embs, return_weights=True)
            self.region_attention = (
                per_fm_region_attention.permute(1, 2, 0, 3, 4)
                * fm_weights.unsqueeze(-1).unsqueeze(-1)
            ).sum(dim=2)
            return logits

        if self.within_slice_agg is not None:
            raise ValueError(
                "regional_tokens > 0 but received global-only inference features. "
                "Provide tiled multi-crop CLS caches with shape "
                "[B, num_slices, num_tiled_regions, input_embed_dim]."
            )

        self.region_attention = None
        return self._pool_fm_embeddings(fm_embs)

    def _abmil_pooling(
        self, 
        h: torch.Tensor, 
        mask: torch.Tensor = None,
        return_per_head: bool = False,
    ) -> torch.Tensor:
        """
        Apply ABMIL attention pooling.
        
        Args:
            h: Input tensor [batch_size, seq_len, embed_dim]
            mask: Boolean mask [batch_size, seq_len] (True = valid)
            return_per_head: If True, return per-head attention [B, num_heads, num_slices]
        
        Returns:
            Attention weights [B, 1, seq_len] or [B, num_heads, seq_len] if return_per_head=True
        """
        if self.num_heads > 1:
            # Split feature dim into heads: [B, num_slices, num_heads, head_dim]
            h_heads = rearrange(h, 'b t (e c) -> b t e c', c=self.num_heads)
            attentions = []
            for i, attn_net in enumerate(self.attn):
                _, raw_attention = attn_net(h_heads[:, :, :, i], mask=mask, return_raw_attention=True) # [B, num_slices, 1]
                attentions.append(raw_attention)
            A = torch.stack(attentions, dim=-1) # [B, num_slices, 1, num_heads]

            if return_per_head:
                per_head = A.squeeze(2).permute(0, 2, 1)  # [B, num_heads, num_slices]
                return F.softmax(per_head, dim=-1)

            A = rearrange(A, 'b t e c -> b t (e c)', c=self.num_heads).mean(-1).unsqueeze(-1) # [B, num_slices, 1]
            A = torch.transpose(A, 2, 1) # [B, 1, num_slices]
            A = F.softmax(A, dim=-1) # [B, 1, num_slices]
        else:
            A = self.attn[0](h, mask=mask) 
            A = torch.transpose(A, 2, 1) # [B, 1, num_slices]
        return A

    def _cross_attention_pooling(
        self,
        h: torch.Tensor,
        mask: torch.Tensor = None,
        values: torch.Tensor = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Apply cross-attention slice pooling.

        In pretraining, keys and values both come from the post-encoder hidden states h by default.
        In inference mode, values are set to an earlier representation (e.g., post-embed features from multiple foundation models)
        while attention scores are still computed from the post-encoder hidden states h (contextualized encoder states).
        """
        if values is None:
            return self.cross_attn_pool(
                h,
                mask=mask,
                return_attention=return_attention,
            )

        _, attn = self.cross_attn_pool(h, mask=mask, return_attention=True)
        pooled = torch.bmm(attn, values).mean(dim=1)
        if return_attention:
            return pooled, attn
        return pooled
    
    def _forward_packed(
        self, 
        x, 
        input_feature_dims: torch.Tensor = None,
        get_attention: bool = False,
        get_region_attention: bool = False,
        return_slice_embeddings: bool = False,
        cu_seqlens: torch.Tensor = None,
        max_seqlen: int = None,
        seq_idx: torch.Tensor = None,
        slice_mask: torch.Tensor = None,
        mask_token: torch.Tensor = None,
        physical_positions: torch.Tensor = None,
        **_,  # Ignore extra kwargs
    ) -> torch.Tensor:
        """Forward pass for packed/variable-length sequences."""
        batch_size = cu_seqlens.shape[0] - 1
        
        # Foundation model feature embedding
        if self.mode == "inference":
            logits = self._embed_inference_forward(x)
        else:
            logits = self._embed_ssl_forward_packed(x, cu_seqlens, input_feature_dims)

        logits = self._collapse_tiled_regions(logits, packed=True)

        if get_region_attention:
            if self.region_attention is None:
                raise ValueError(
                    "Region attention is only available for tiled multi-crop CLS models "
                    "(regional_tokens > 0)."
                )
            return [
                self.region_attention[
                    0,
                    int(cu_seqlens[i].item()):int(cu_seqlens[i + 1].item()),
                ]
                for i in range(batch_size)
            ]
        
        # Physical positional encoding (packed: positions is 1-D [total_seq_len])
        if self.physical_pe and physical_positions is not None:
            logits = self._apply_physical_pe(logits, physical_positions)

        # MSP: replace masked positions with mask_token
        if slice_mask is not None and mask_token is not None:
            logits = torch.where(
                slice_mask.unsqueeze(-1),
                mask_token.expand_as(logits),
                logits,
            )
        
        # Sequence encoder
        if self.sequence_encoder == "transformer" and hasattr(self, 'varlen_seq_enc'):
            # Transformer with FlashAttention varlen
            # Processes all sequences in one pass without padding
            h = self.varlen_seq_enc(logits, cu_seqlens, max_seqlen)
            # Note: VarlenTransformerEncoder already has final LayerNorm
        else:
            # Pack all sequences into one batch with seq_idx indicating sequence boundaries
            logits_packed = logits.unsqueeze(0)  # [1, total_seq_len, embed_dim]
            seq_idx_2d = seq_idx.unsqueeze(0) if seq_idx is not None and seq_idx.dim() == 1 else seq_idx # [1, total_seq_len]
            h = self.seq_enc(logits_packed, seq_idx=seq_idx_2d).squeeze(0)  # [total_seq_len, embed_dim]
            h = self.norm(h)
        
        if return_slice_embeddings:
            return h
        
        # Slice feature aggregation
        h_padded, mask = self._packed_to_padded(h, cu_seqlens, max_seqlen)

        if self.slice_pooling == "cross_attention":
            if get_attention:
                _, attn = self._cross_attention_pooling(
                    h_padded,
                    mask=mask,
                    return_attention=True,
                )
                return [
                    attn[i, :, :int((cu_seqlens[i + 1] - cu_seqlens[i]).item())]
                    for i in range(batch_size)
                ]
            if self.mode == "train" or self.pooling_target == "post_encoder":
                pooled = self._cross_attention_pooling(h_padded, mask=mask)
            elif self.pooling_target == "post_embed":
                logits_padded, _ = self._packed_to_padded(logits, cu_seqlens, max_seqlen)
                pooled = self._cross_attention_pooling(
                    h_padded,
                    mask=mask,
                    values=logits_padded,
                )
            else:
                raise ValueError(
                    "slice_pooling='cross_attention' supports pooling_target "
                    "'post_encoder' or 'post_embed'. Raw pooling is only supported by ABMIL."
                )
            return self.proj(pooled) if self.mode == "train" else pooled
        
        # ABMIL path
        A = self._abmil_pooling(h_padded, mask)
        
        if get_attention:
            return [
                A[i, :, :int((cu_seqlens[i + 1] - cu_seqlens[i]).item())]
                for i in range(batch_size)
            ]
        
        if self.mode == "train":
            pooled = torch.bmm(A, h_padded).squeeze(1)
            return self.proj(pooled)
        if self.pooling_target == "post_encoder":
            return torch.bmm(A, h_padded).squeeze(1)        # [B, embed_dim]
        elif self.pooling_target == "raw":
            if isinstance(x, list):
                logger.info("Multi-FM mode: using first FM embedding dimension for raw pooling")
                raw_x = x[0]
            else:
                raw_x = x
            raw_padded, _ = self._packed_to_padded(raw_x, cu_seqlens, max_seqlen)
            return torch.bmm(A, raw_padded).squeeze(1)    # [B, raw_output_dim]
        else:  # post_embed
            logits_padded, _ = self._packed_to_padded(logits, cu_seqlens, max_seqlen)
            return torch.bmm(A, logits_padded).squeeze(1)   # [B, embed_dim]
        

    def _forward(
        self, 
        x, 
        input_feature_dims=None, 
        get_attention=False, 
        get_per_head_attention=False,
        return_slice_embeddings=False,
        get_region_attention=False,
        seq_lengths=None,
        slice_mask=None,
        mask_token=None,
        physical_positions=None,
        **_,
    ):
        """Forward pass main function."""
        # Foundation model feature embedding
        if self.mode == "inference":
            logits = self._embed_inference_forward(x, seq_lengths=seq_lengths)
        else:
            logits = self._embed_ssl_forward(x, input_feature_dims)

        logits = self._zero_padded_slices(logits, seq_lengths)

        if self.mode != "inference" or logits.dim() == 4:
            logits = self._collapse_tiled_regions(logits, packed=False)

        # Region-level (within-slice) attention for quadrant interpretability.
        if get_region_attention:
            if self.region_attention is None:
                raise ValueError(
                    "Region attention is only available for tiled multi-crop CLS models "
                    "(regional_tokens > 0)."
                )
            return self.region_attention  # [B, num_slices, 1, num_tiled_regions-1]

        # Physical positional encoding (padded: positions is [B, num_slices])
        if self.physical_pe and physical_positions is not None:
            logits = self._apply_physical_pe(logits, physical_positions, seq_lengths)

        # MSP: replace masked positions with mask_token
        if slice_mask is not None and mask_token is not None:
            logits = torch.where(
                slice_mask.unsqueeze(-1).expand_as(logits),
                mask_token.unsqueeze(0).expand_as(logits),
                logits,
            )

        # Build attention mask if seq_lengths is provided
        mask = None
        if seq_lengths is not None:
            max_len = logits.shape[1]  
            mask = self._build_mask(seq_lengths, max_len)  # [B, num_slices]

        # Prepend CLS token for cls pooling
        if self.slice_pooling == "cls":
            B = logits.shape[0]
            logits = torch.cat([self.cls_token.expand(B, -1, -1), logits], dim=1)

        # Build src_key_padding_mask for transformer
        src_key_padding_mask = None
        if mask is not None and self.sequence_encoder == "transformer":
            if self.slice_pooling == "cls":
                cls_mask = torch.zeros((mask.shape[0], 1), dtype=torch.bool, device=mask.device)
                src_key_padding_mask = torch.cat([cls_mask, ~mask], dim=1)
            else:
                src_key_padding_mask = ~mask

        # Sequence encoder
        if self.sequence_encoder == "mamba2":
            h = self.seq_enc(logits)
        else:
            h = self.seq_enc(logits, src_key_padding_mask=src_key_padding_mask)
        h = self.norm(h)

        # Return slice-level embeddings before aggregation
        if return_slice_embeddings:
            if self.slice_pooling == "cls":
                return h[:, 1:, :]  # Remove CLS token
            return h # [B, num_slices, embed_dim]

        # Slice feature aggregation
        # CLS token pooling
        if self.slice_pooling == "cls":
            if get_attention or get_per_head_attention:
                return self._extract_cls_attention(logits, mask, src_key_padding_mask)
            pooled = h[:, 0, :]
            return self.proj(pooled) if self.mode == "train" else pooled

        # Cross-attention pooling
        if self.slice_pooling == "cross_attention":
            if get_attention or get_per_head_attention:
                _, attn = self._cross_attention_pooling(
                    h,
                    mask=mask,
                    return_attention=True,
                )
                return attn  # [B, 1, num_slices]
            if self.mode == "train" or self.pooling_target == "post_encoder":
                pooled = self._cross_attention_pooling(h, mask=mask)
            elif self.pooling_target == "post_embed":
                pooled = self._cross_attention_pooling(h, mask=mask, values=logits)
            else:
                raise ValueError(
                    "slice_pooling='cross_attention' supports pooling_target "
                    "'post_encoder' or 'post_embed'. Raw pooling is only supported by ABMIL."
                )
            return self.proj(pooled) if self.mode == "train" else pooled

        # ABMIL pooling
        if get_per_head_attention:
            return self._abmil_pooling(h, mask, return_per_head=True)

        A = self._abmil_pooling(h, mask)

        if get_attention:
            return A
        
        # Attention-weighted aggregation
        if self.mode == "train":
            pooled = torch.bmm(A, h).squeeze(1)  # [B, embed_dim]
            return self.proj(pooled)

        if self.pooling_target == "post_encoder":
            return torch.bmm(A, h).squeeze(1)  # [B, embed_dim]
        elif self.pooling_target == "raw":
            if isinstance(x, list):
                logger.info("Multi-FM mode: using first FM embedding dimension for raw pooling")
                raw_x = x[0]
            else:
                raw_x = x
            return torch.bmm(A, raw_x).squeeze(1)  # [B, raw_output_dim]
        else:  # post_embed
            return torch.bmm(A, logits).squeeze(1)  # [B, embed_dim]

    def forward(
        self, 
        x, 
        *,
        input_feature_dims=None, 
        get_attention=False, 
        get_per_head_attention=False,
        return_slice_embeddings=False,
        get_region_attention=False,
        use_packed: bool = False,
        slice_mask=None,
        mask_token=None,
        physical_positions=None,
        **kwargs,
    ):
        """
        Forward pass through the Cobra network.
        
        Args:
            x: Input tensor or list of tensors:
                - SSL padded, global-only: [B, num_slices, input_embed_dim]
                - SSL padded, tiled: [B, num_slices, num_tiled_regions, input_embed_dim]
                - SSL packed, global-only: [total_slices, input_embed_dim]
                - SSL packed, tiled: [total_slices, num_tiled_regions, input_embed_dim]
                - Inference, global-only: List of K tensors, each [B, num_slices, input_embed_dim]
                - Inference, tiled: List of K tensors, each [B, num_slices, num_tiled_regions, input_embed_dim]
            input_feature_dims: Feature dimensions per sample [B] (SSL mode).
            get_attention: If True, return aggregated attention map [B, 1, num_slices].
            get_per_head_attention: If True, return per-head attention [B, num_heads, num_slices] (ABMIL only).
            return_slice_embeddings: If True, return slice-level embeddings [B, num_slices, embed_dim] before pooling.
            use_packed: If True, use packed sequence for variable sequence length handling.
            slice_mask: Optional boolean mask for MSP. True = masked.
                - Padded mode: [B, num_slices]
                - Packed mode: [total_seq_len]
            mask_token: Learnable mask-token embedding [1, embed_dim] (for masked slice prediction).
            physical_positions: Physical slice positions in mm for sinusoidal PE.
                - Padded mode: [B, num_slices]
                - Packed mode: [total_seq_len]
                Only used when `physical_pe=True`.
            **kwargs: Mode-specific parameters:
                Padded mode:
                    - seq_lengths: Actual sequence lengths [B] for masking padded positions.
                Packed mode:
                    - cu_seqlens: Cumulative sequence lengths [B+1], int32
                    - max_seqlen: Maximum sequence length in the batch
                    - seq_idx: Document index for each token [total_seq_len], int32
        
        Returns:
            If get_attention=True: Attention map [B, 1, num_slices]
            If get_per_head_attention=True: Per-head attention [B, num_heads, num_slices]
            If return_slice_embeddings=True: Slice embeddings [B, num_slices, embed_dim]
            Otherwise: Features [B, contrast_dim] (train) or [B, output_dim] (inference)
        """
        if use_packed:
            if kwargs.get('cu_seqlens') is None or kwargs.get('max_seqlen') is None or kwargs.get('seq_idx') is None:
                raise ValueError("cu_seqlens, max_seqlen, and seq_idx are required for packed mode.")
            return self._forward_packed(
                x,
                input_feature_dims=input_feature_dims,
                get_attention=get_attention,
                get_region_attention=get_region_attention,
                return_slice_embeddings=return_slice_embeddings,
                slice_mask=slice_mask,
                mask_token=mask_token,
                physical_positions=physical_positions,
                **kwargs,
            )
        else:
            return self._forward(
                x,
                input_feature_dims=input_feature_dims,
                get_attention=get_attention,
                get_per_head_attention=get_per_head_attention,
                return_slice_embeddings=return_slice_embeddings,
                get_region_attention=get_region_attention,
                slice_mask=slice_mask,
                mask_token=mask_token,
                physical_positions=physical_positions,
                **kwargs,
            )