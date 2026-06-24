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
from .fm_router import FMRouter
from .positional_encoding import sinusoidal_position_encoding
from med_slim.model.attention_pooling import (
    BatchedABMIL,
    InterSliceAggregator,
)
from med_slim.logging.setup import init_logging

init_logging()
logger = logging.getLogger(__name__)

POOLING_TARGETS = ("post_encoder", "post_embed", "raw")


def _resolve_pooling_target(
    mode: str,
    pooling_target: Optional[str],
    regional_tokens: int = 0,
    slice_pooling: str = "abmil",
) -> Optional[str]:
    if mode not in ("train", "inference"):
        raise ValueError(f"Invalid mode '{mode}'. Must be 'train' or 'inference'.")
    if slice_pooling not in ("abmil", "cls"):
        raise ValueError(
            f"Invalid slice_pooling '{slice_pooling}'. Must be 'abmil' or 'cls'."
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
            "raw"
            if slice_pooling == "abmil" and regional_tokens == 0
            else "post_embed"
        )
    return resolved


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
        num_fms: Optional[int] = None,
        per_fm_adapter_mode: str = "per_dim",
        fm_input_dims: Optional[List[int]] = None,
        router_use_fm_embedding: bool = True,
        router_use_fm_logit_bias: bool = True,
        router_use_fm_logit_scale: bool = False,
        router_mode: str = "soft",
        router_top_k: Optional[int] = None,
        router_temperature: float = 1.0,
        router_learnable_temperature: bool = False,
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
            fm_pooling: 'avg_pool' or 'router'.
                - 'avg_pool': Uniform average pooling over the selected foundation models.
                - 'router': Variable-K soft per-slice FMRouter (MoE-style gating).
            slice_pooling: 'abmil' or 'cls' (cls requires transformer).
            pooling_target: Which representation level to aggregate at inference.
                Ignored in training mode, where Cobra always pools post-encoder hidden
                states for contrastive pretraining. When None, inference defaults are
                resolved by ``_resolve_pooling_target`` (see that helper for per-path
                defaults). Meaning depends on ``slice_pooling``:
                - 'abmil': ABMIL attention weights come from encoder output; pooling_target
                  selects which features they aggregate ('raw', 'post_embed', or
                  'post_encoder'). For tiled features, 'raw' pools flattened global
                  and regional FM tokens.
                - 'cls': not applicable, volume embedding is always the
                  sequence-encoder CLS token. Resolved pooling_target is None
            raw_output_dim: FM embedding dimension for the 'raw' pooling target.
                Required in inference mode when the resolved pooling_target is 'raw'.
            physical_pe: If True, add sinusoidal positional encoding based on normalized relative slice depth in [0, 1] before the sequence encoder.
                Requires ``physical_positions`` to be passed during forward.
            regional_tokens: Number of regional crop tokens per slice for tiled multi-crop CLS (e.g. 4 for a 2x2 grid). 0 (default) disables tiled
                mode and keeps the global-only CLS pathway. When > 0, [B, num_slices, 1+regional_tokens, embed_dim]
                tokens are flattened to [B, num_slices*(1+regional_tokens), embed_dim] before the sequence encoder.
            num_fms: Total number of foundation models. Required when fm_pooling='router'.
            per_fm_adapter_mode: Projection-adapter keying for the FM-set paths (subset SSL and
                inference FM-set). 'per_dim' shares one Embed MLP across FMs of the same input
                dim. 'per_fm_id' builds one Embed MLP per global FM id so each FM is a distinct expert. 
                Pair SSL and all packed paths always use the dim-keyed adapters based on COBRA approach.
            fm_input_dims: Input feature dim per global FM id (ordered by global FM id). Required
                when per_fm_adapter_mode='per_fm_id'; used to size the per-FM Embed adapters.
            router_use_fm_embedding: Add learned per-FM identity embedding before scoring.
            router_use_fm_logit_bias: Add learned per-FM additive logit bias to each FM token before scoring.
            router_use_fm_logit_scale: Add learned per-FM positive logit scale factor to each FM token before scoring.
            router_mode: 'soft' (dense softmax) or 'topk' (sparse top-k softmax).
            router_top_k: Number of FMs kept when router_mode='topk'.
            router_temperature: Router softmax temperature.
            router_learnable_temperature: If True, the router temperature is a learned parameter.
            **kwargs: Additional encoder-specific parameters:
                - d_state: Mamba2 state dimension (default: 128)
                - dim_feedforward: Transformer FFN dimension (default: 4*embed_dim)
                - norm_first: Pre-LN transformer (default: True)
                - rotary_positional_encoding: RoPE config (default: None)
                - att_dim: ABMIL hidden dimension for slice pooling (default: 256)
        """
        super().__init__()

        if input_dims is None:
            input_dims = [512, 768, 1024, 1152, 1376, 1536]

        assert mode in ["train", "inference"]
        assert sequence_encoder in ["mamba2", "transformer"]
        assert fm_pooling in ["avg_pool", "router"], (
            f"Invalid fm_pooling '{fm_pooling}'. Must be one of 'avg_pool', 'router'."
        )
        assert per_fm_adapter_mode in ["per_dim", "per_fm_id"], (
            f"Invalid per_fm_adapter_mode '{per_fm_adapter_mode}'. "
            f"Must be one of 'per_dim', 'per_fm_id'."
        )
        assert slice_pooling in ["abmil", "cls"], (
            f"Invalid slice_pooling '{slice_pooling}'. Must be one of 'abmil', 'cls'."
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
            and slice_pooling != "cls"
            and raw_output_dim is None
        ):
            raise ValueError("raw_output_dim is required when pooling_target='raw' in inference mode.")

        self.mode = mode
        self.embed_dim = embed_dim
        self.pooling_target = resolved_pooling_target
        self._raw_output_dim = raw_output_dim
        self.sequence_encoder = sequence_encoder
        self.fm_pooling = fm_pooling
        self.num_fms = num_fms
        self.slice_pooling = slice_pooling
        self.physical_pe = physical_pe

        # Tiled multi-crop CLS: keep global + regional tokens as sequence tokens.
        self.regional_tokens = regional_tokens

        # FM-set projection adapters.
        self._last_fm_stats = None # For router load-balance/z-loss
        self.per_fm_adapter_mode = per_fm_adapter_mode
        self.fm_input_dims = list(fm_input_dims) if fm_input_dims is not None else None
        self.embed = None
        self.embed_fm = None
        if per_fm_adapter_mode == "per_dim":
            self.embed = nn.ModuleDict({str(d): Embed(d, embed_dim) for d in input_dims})
        else:
            if fm_input_dims is None:
                raise ValueError(
                    "per_fm_adapter_mode='per_fm_id' requires fm_input_dims "
                    "(input dim per global FM id)."
                )
            if num_fms is not None and len(fm_input_dims) != num_fms:
                raise ValueError(
                    f"fm_input_dims length ({len(fm_input_dims)}) must equal num_fms "
                    f"({num_fms})."
                )
            self.embed_fm = nn.ModuleDict(
                {str(i): Embed(d, embed_dim) for i, d in enumerate(fm_input_dims)}
            )

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
        
        # FM fusion modules: weight/select different foundation model embeddings at each slice.
        self.fm_router = None
        if fm_pooling == "router":
            if num_fms is None:
                raise ValueError(
                    "fm_pooling='router' requires num_fms (total number of FMs)."
                )
            self.fm_router = FMRouter(
                embed_dim=embed_dim,
                num_fms=num_fms,
                use_fm_embedding=router_use_fm_embedding,
                use_fm_logit_bias=router_use_fm_logit_bias,
                use_fm_logit_scale=router_use_fm_logit_scale,
                dropout=dropout,
                router_mode=router_mode,
                top_k=router_top_k,
                temperature=router_temperature,
                learnable_temperature=router_learnable_temperature,
            )

    def _apply_physical_pe(
        self,
        logits: torch.Tensor,
        physical_positions: torch.Tensor,
        seq_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Add sinusoidal PE keyed on relative slice depth in [0, 1] to embedded features.

        PE at padded positions is zeroed out so that padding remains inert.
        Without this, PE(0) = [0, 1, 0, 1, ...] (cos(0)=1) would inject a
        non-zero signal into padded positions and make relative depth 0 ambiguous
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

    def _per_dim_adapter(self, feat_dim: int) -> Embed:
        """Return a dim-keyed adapter for FM-set, or fail clearly when it is not instantiated."""
        if self.embed is None:
            raise ValueError(
                "Per-dim projection adapters are disabled "
                "because per_fm_adapter_mode='per_fm_id'. "
                "This mode only supports FM-set inputs with fm_ids (router subset SSL or FM-set inference). "
                "Use per_fm_adapter_mode='per_dim' for pair SSL or packed dim-keyed inputs."
            )
        return self.embed[str(feat_dim)]

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
            logits = torch.concat([self._per_dim_adapter(input_feature_dims[i].item())(x[i][..., :input_feature_dims[i].item()]).unsqueeze(0) for i in range(len(x))], dim=0) # [B, num_slices, (num_tiled_regions,) embed_dim]
        else:
            logits = self._per_dim_adapter(x.shape[-1])(x)  # [B, num_slices, (num_tiled_regions,) embed_dim]
        return logits

    def _embed_ssl_forward_packed_global_only(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        input_feature_dims: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed packed global-only CLS features [total_slices, input_embed_dim]."""
        if input_feature_dims is None:
            return self._per_dim_adapter(x.shape[-1])(x)  # [total_slices, embed_dim]

        input_feature_dims = input_feature_dims.to(device=x.device, dtype=torch.long)
        total_seq_len = x.shape[0]
        seq_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(device=x.device)  # [B]
        unique_dims = input_feature_dims.unique()

        if len(unique_dims) == 1:
            feat_dim = unique_dims[0].item()
            return self._per_dim_adapter(feat_dim)(x[:, :feat_dim])

        slice_feat_dims = torch.repeat_interleave(input_feature_dims, seq_lens)  # [total_slices]
        logits = None
        for dim in unique_dims:
            feat_dim = dim.item()
            mask = slice_feat_dims == dim  # [total_slices]
            x_group = x[mask, :feat_dim]  # [num_slices_for_dim, input_embed_dim]
            embedded = self._per_dim_adapter(feat_dim)(x_group)  # [num_slices_for_dim, embed_dim]
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
            return self._per_dim_adapter(x.shape[-1])(x)  # [total_slices, num_tiled_regions, embed_dim]

        input_feature_dims = input_feature_dims.to(device=x.device, dtype=torch.long)
        total_seq_len, num_tiled_regions = x.shape[:2]
        seq_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(device=x.device)  # [B]
        unique_dims = input_feature_dims.unique()

        if len(unique_dims) == 1:
            feat_dim = unique_dims[0].item()
            return self._per_dim_adapter(feat_dim)(x[..., :feat_dim])

        slice_feat_dims = torch.repeat_interleave(input_feature_dims, seq_lens)  # [total_slices]
        logits = None
        for dim in unique_dims:
            feat_dim = dim.item()
            mask = slice_feat_dims == dim  # [total_slices]
            x_group = x[mask, :, :feat_dim]  # [num_slices_for_dim, num_tiled_regions, input_embed_dim]
            embedded = self._per_dim_adapter(feat_dim)(x_group)  # [num_slices_for_dim, num_tiled_regions, embed_dim]
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

    def _validate_num_regions(self, num_regions: int) -> None:
        expected = 1 + self.regional_tokens
        if self.regional_tokens <= 0:
            raise ValueError(
                "Received tiled regional-token features, but regional_tokens=0. "
                "Set regional_tokens to the number of regional crop tokens."
            )
        if num_regions != expected:
            raise ValueError(
                f"Expected {expected} tokens per slice "
                f"(1 global + {self.regional_tokens} regional), got {num_regions}."
            )

    @staticmethod
    def _expand_lengths_for_regions(
        seq_lengths: torch.Tensor | None,
        num_regions: int,
    ) -> torch.Tensor | None:
        if seq_lengths is None:
            return None
        return seq_lengths * num_regions

    @staticmethod
    def _expand_physical_positions_for_regions(
        physical_positions: torch.Tensor | None,
        num_regions: int,
    ) -> torch.Tensor | None:
        if physical_positions is None:
            return None
        return physical_positions.repeat_interleave(num_regions, dim=-1)

    def _flatten_regional_tokens(
        self,
        x: torch.Tensor,
        *,
        fm_set_layout: bool = False,
        packed: bool = False,
        seq_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor | None, int]:
        """
        Flatten regional tokens into the sequence dimension without learned aggregation.

        `fm_set_layout` is controlled by tensor layout:
        - `fm_set_layout=False`: single-FM view without leading K axis. Global-only
          shapes are `[B, D, E]` or packed `[T, E]`.
        - `fm_set_layout=True`: FM-set layout with a leading K axis. Global-only
          shape is `[K, B, D, E]`.
        """
        if fm_set_layout:
            if x.dim() == 4:
                return x, seq_lengths, 1
            if x.dim() != 5:
                raise ValueError(
                    f"FM-set inputs must be rank 4 (global-only) or rank 5 "
                    f"(regional tokens), got rank {x.dim()}."
                )
            num_fms, batch_size, num_slices, num_regions, embed_dim = x.shape
            self._validate_num_regions(num_regions)
            flat = x.reshape(num_fms, batch_size, num_slices * num_regions, embed_dim)
            return flat, self._expand_lengths_for_regions(seq_lengths, num_regions), num_regions

        tiled_rank = 3 if packed else 4
        global_rank = 2 if packed else 3

        if x.dim() == tiled_rank:
            num_regions = x.shape[1] if packed else x.shape[2]
            self._validate_num_regions(num_regions)
            if packed:
                total_slices, _, embed_dim = x.shape
                flat = x.reshape(total_slices * num_regions, embed_dim)
            else:
                batch_size, num_slices, _, embed_dim = x.shape
                flat = x.reshape(batch_size, num_slices * num_regions, embed_dim)
            return flat, self._expand_lengths_for_regions(seq_lengths, num_regions), num_regions

        if x.dim() != global_rank:
            raise ValueError(
                f"Expected global-only rank {global_rank}, got rank {x.dim()}."
            )
        return x, seq_lengths, 1

    def _embed_subset_fm_set(
        self,
        x: torch.Tensor,
        input_feature_dims: torch.Tensor,
        fm_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Embed a subset-SSL FM set into the FM-set layout.

        Args:
            x: Subset global-only features [B, K, num_slices, input_embed_dim] or
               subset tiled features [B, K, num_slices, num_tiled_regions, input_embed_dim].
            input_feature_dims: Original feature dims per (sample, FM) with shape [B, K].
            fm_ids: Global FM IDs per (sample, FM) with shape [B, K]. Required when
                per_fm_adapter_mode='per_fm_id' to select the per-FM adapter.

        Returns:
            FM-set tensor [K, B, num_slices, (num_tiled_regions,) embed_dim]
        """
        if input_feature_dims is None:
            raise ValueError(
                "Subset SSL embedding requires input_feature_dims with shape [B, K]."
            )
        batch_size, num_fms = x.shape[0], x.shape[1]
        spatial = x.shape[2:-1]  # (num_slices,) or (num_slices, num_tiled_regions)
        x_flat = x.reshape(batch_size * num_fms, *x.shape[2:])  # [B * K, ..., input_embed_dim]
        dims_flat = input_feature_dims.reshape(-1).to(device=x.device, dtype=torch.long)
        assert dims_flat.shape[0] == x_flat.shape[0], (
            f"input_feature_dims must have {x_flat.shape[0]} entries (B*K), "
            f"got {dims_flat.shape[0]}."
        )

        out = None
        if self.per_fm_adapter_mode == "per_fm_id":
            # Group (sample, FM) entries by global FM id so each FM uses its own adapter.
            if fm_ids is None:
                raise ValueError(
                    "per_fm_adapter_mode='per_fm_id' requires fm_ids with shape [B, K]."
                )
            ids_flat = fm_ids.reshape(-1).to(device=x.device, dtype=torch.long)
            assert ids_flat.shape[0] == x_flat.shape[0], (
                f"fm_ids must have {x_flat.shape[0]} entries (B*K), "
                f"got {ids_flat.shape[0]}."
            )
            for fid in ids_flat.unique():
                fm_id = int(fid.item())
                feat_dim = int(self.fm_input_dims[fm_id])
                group_mask = ids_flat == fid
                embedded = self.embed_fm[str(fm_id)](x_flat[group_mask][..., :feat_dim])
                if out is None:
                    out = x_flat.new_zeros(batch_size * num_fms, *spatial, self.embed_dim)
                out[group_mask] = embedded.to(out.dtype)
        else:
            # Group (sample, FM) entries by feature dim because Embed is keyed by dim and
            # multiple FMs can share the same dimension.
            for dim in dims_flat.unique():
                feat_dim = int(dim.item())
                group_mask = dims_flat == dim
                embedded = self._per_dim_adapter(feat_dim)(x_flat[group_mask][..., :feat_dim])
                if out is None:
                    out = x_flat.new_zeros(batch_size * num_fms, *spatial, self.embed_dim)
                out[group_mask] = embedded.to(out.dtype)

        out = out.reshape(batch_size, num_fms, *spatial, self.embed_dim)  # [B, K, ..., E]
        perm = [1, 0] + list(range(2, out.dim()))  # -> [K, B, ..., E]
        return out.permute(*perm).contiguous()

    def _fuse_fm_embeddings(
        self,
        fm_embs: torch.Tensor,
        fm_ids: torch.Tensor | None = None,
        return_fm_stats: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """
        Fuse post-Embed, region-collapsed representations across foundation models.

        Args:
            fm_embs: [num_fms, B, num_slices, embed_dim].
            fm_ids: Global FM IDs [B, num_fms] for the selected FMs (router/logging).
            return_fm_stats: If True, return an FM statistics dict additionally.

        Returns:
            fused: [B, num_slices, embed_dim].
            stats (optional): dict with fm_weights_local [B, num_slices, num_fms],
                fm_ids [B, num_fms], router_logits [B, num_slices, num_fms] or None,
                fm_weight_entropy [B, num_slices].
        """
        num_fms, batch_size, num_slices, embed_dim = fm_embs.shape
        if fm_ids is None:
            fm_ids = torch.arange(num_fms, device=fm_embs.device).unsqueeze(0).expand(
                batch_size, -1
            )
        else:
            fm_ids = fm_ids.to(device=fm_embs.device, dtype=torch.long)
            if fm_ids.dim() == 1:
                if fm_ids.numel() != num_fms:
                    raise ValueError(
                        f"Expected {num_fms} FM IDs for an FM set, got {fm_ids.numel()}."
                    )
                fm_ids = fm_ids.unsqueeze(0).expand(batch_size, -1)
            elif fm_ids.shape == (1, num_fms) and batch_size != 1:
                fm_ids = fm_ids.expand(batch_size, -1)
            elif fm_ids.shape != (batch_size, num_fms):
                raise ValueError(
                    f"Expected fm_ids shape ({batch_size}, {num_fms}) or ({num_fms},), "
                    f"got {tuple(fm_ids.shape)}."
                )

        if self.fm_pooling == "router":
            if self.fm_router is None:
                raise ValueError("fm_pooling='router' but FMRouter is not initialized.")
            fused, stats = self.fm_router(
                fm_embs,
                fm_ids=fm_ids,
                return_stats=True,
            )
            if return_fm_stats:
                return fused, stats
            return fused

        if num_fms == 1:
            fused = fm_embs[0]
            weights = torch.ones(
                batch_size, num_slices, 1, device=fm_embs.device, dtype=fm_embs.dtype
            )
        elif self.fm_pooling == "avg_pool":
            fused = fm_embs.mean(dim=0)
            weights = torch.full(
                (batch_size, num_slices, num_fms),
                1.0 / num_fms,
                device=fm_embs.device,
                dtype=fm_embs.dtype,
            )
        else:
            raise ValueError(f"Unsupported fm_pooling '{self.fm_pooling}'.")

        if return_fm_stats:
            probs = weights.clamp_min(1e-8)
            stats = {
                "fm_weights_local": weights,
                "fm_ids": fm_ids,
                "router_logits": None,
                "fm_weight_entropy": -(probs * probs.log()).sum(dim=-1),
            }
            return fused, stats
        return fused

    def _fuse_fm_set(
        self,
        fm_embs: torch.Tensor,
        fm_ids: torch.Tensor | None = None,
        seq_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, int]:
        """
        Aggregate a set of FM embeddings to one fused representation per slice.

        Args:
            fm_embs: [K, B, num_slices, (num_regions,) embed_dim]
            fm_ids: Global FM IDs [B, K] (router/logging).
            seq_lengths: Optional [B] for masking padded slices in router stats.

        Returns:
            fused token embeddings [B, num_tokens, embed_dim], expanded sequence
            lengths, and number of tokens per original slice.
        """
        fm_slice_embs, token_lengths, num_regions = self._flatten_regional_tokens(
            fm_embs, fm_set_layout=True, seq_lengths=seq_lengths
        )
        fused, fm_stats = self._fuse_fm_embeddings(
            fm_slice_embs, fm_ids=fm_ids, return_fm_stats=True
        )
        if token_lengths is not None:
            fm_stats["slice_mask"] = self._build_mask(token_lengths, fm_slice_embs.shape[2])
        self._last_fm_stats = fm_stats

        return fused, token_lengths, num_regions
        
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
        """Zero padded slice or flattened slice-region token positions."""
        if seq_lengths is None:
            return logits
        max_len = logits.shape[1]
        mask = self._build_mask(seq_lengths, max_len)
        if logits.dim() == 3:
            return logits * mask.unsqueeze(-1)
        if logits.dim() == 4:
            return logits * mask.unsqueeze(-1).unsqueeze(-1)
        return logits
            
    def _embed_inference_fm_set(self, x, fm_ids: torch.Tensor | None = None) -> torch.Tensor:
        """
        Embed an inference FM set into the canonical FM-set layout.

        Args:
            x: List of K FM tensors (K may be 1), each [B, num_slices, input_embed_dim]
               (global-only CLS) or [B, num_slices, num_tiled_regions, input_embed_dim]
               (tiled multi-crop CLS).
            fm_ids: Global FM IDs for the K FMs, shape [K] or [B, K]. Required when
                per_fm_adapter_mode='per_fm_id' to select the per-FM adapter.

        Returns:
            FM-set tensor [K, B, num_slices, embed_dim] or
            [K, B, num_slices, num_tiled_regions, embed_dim].
        """
        if self.per_fm_adapter_mode == "per_fm_id":
            if fm_ids is None:
                raise ValueError(
                    "per_fm_adapter_mode='per_fm_id' requires fm_ids to select per-FM adapters."
                )
            # FM identity is shared across the batch at inference; collapse [B, K] -> [K].
            ids = fm_ids if fm_ids.dim() == 1 else fm_ids[0]
            assert ids.shape[0] == len(x), (
                f"fm_ids must have one entry per FM ({len(x)}), got {ids.shape[0]}."
            )
            embedded_features = []
            for i, xi in enumerate(x):
                fm_id = int(ids[i].item())
                feat_dim = int(self.fm_input_dims[fm_id])
                embedded_features.append(self.embed_fm[str(fm_id)](xi[..., :feat_dim]))
        else:
            embedded_features = [self._per_dim_adapter(xi.shape[-1])(xi) for xi in x]  # List of K [B, num_slices, (num_tiled_regions,) embed_dim]
        fm_embs = torch.stack(embedded_features, dim=0)  # [K, B, num_slices, (num_tiled_regions,) embed_dim]
        assert fm_embs.shape[-1] == self.embed_dim, f"Expected embed_dim {self.embed_dim}, got {fm_embs.shape[-1]}"
        assert fm_embs.dim() in (4, 5), f"Expected 4 or 5 dimensions, got {fm_embs.dim()}"
        assert fm_embs.shape[0] == len(x), f"Expected length of input x {len(x)}, got {fm_embs.shape[0]}"
        return fm_embs

    def _prepare_pair_ssl_view(
        self,
        x: torch.Tensor,
        input_feature_dims: torch.Tensor | None,
        seq_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, int]:
        """
        Pair SSL (single-FM view): embed one FM tensor and aggregate tiled regions.

        Input: [B, num_slices, (num_tiled_regions,) input_embed_dim].
        Output: [B, num_tokens, embed_dim] with num_tokens=num_slices*num_regions.
        """
        logits = self._embed_ssl_forward(x, input_feature_dims)  # [B, D, (R,) E]
        logits = self._zero_padded_slices(logits, seq_lengths)
        return self._flatten_regional_tokens(logits, fm_set_layout=False, seq_lengths=seq_lengths)

    def _prepare_subset_fm_set(
        self,
        x: torch.Tensor,
        input_feature_dims: torch.Tensor | None,
        fm_ids: torch.Tensor,
        seq_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, int]:
        """
        Subset SSL (FM set per view): embed the FM set and fuse across FMs.

        Input: [B, K, num_slices, (num_tiled_regions,) input_embed_dim] plus fm_ids [B, K].
        Output: [B, num_tokens, embed_dim] with num_tokens=num_slices*num_regions.
        """
        fm_embs = self._embed_subset_fm_set(x, input_feature_dims, fm_ids=fm_ids)  # [K, B, D, (R,) E]
        logits, token_lengths, num_regions = self._fuse_fm_set(
            fm_embs, fm_ids=fm_ids, seq_lengths=seq_lengths
        )
        return self._zero_padded_slices(logits, token_lengths), token_lengths, num_regions

    def _prepare_inference_fm_set(
        self,
        x,
        seq_lengths: torch.Tensor | None = None,
        fm_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, int]:
        """
        Inference (single- or multi-FM): embed the FM set and fuse across FMs.

        Single-FM inference is the K=1 case of an FM set, so it shares this path.

        Input: list of K FM tensors, each [B, num_slices, (num_tiled_regions,) input_embed_dim].
        Output: [B, num_tokens, embed_dim] with num_tokens=num_slices*num_regions.
        """
        fm_embs = self._embed_inference_fm_set(x, fm_ids=fm_ids)  # [K, B, D, (R,) E]
        logits, token_lengths, num_regions = self._fuse_fm_set(
            fm_embs, fm_ids=fm_ids, seq_lengths=seq_lengths
        )
        return self._zero_padded_slices(logits, token_lengths), token_lengths, num_regions

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
    
    def _forward_packed(
        self, 
        x, 
        input_feature_dims: torch.Tensor = None,
        get_attention: bool = False,
        return_slice_embeddings: bool = False,
        cu_seqlens: torch.Tensor = None,
        max_seqlen: int = None,
        seq_idx: torch.Tensor = None,
        physical_positions: torch.Tensor = None,
        **_,  # Ignore extra kwargs
    ) -> torch.Tensor:
        """Forward pass for packed/variable-length sequences."""
        batch_size = cu_seqlens.shape[0] - 1
        num_regions = self._get_num_tiled_regions(x, packed=True)
        
        # Foundation model feature embedding
        if self.mode == "inference":
            # Inference FM set (K=1 when in single-FM inference mode) is aggregated to [B, num_slices, embed_dim].
            logits, _, _ = self._prepare_inference_fm_set(x)
        else:
            logits = self._embed_ssl_forward_packed(x, cu_seqlens, input_feature_dims)
            logits, _, num_regions = self._flatten_regional_tokens(logits, packed=True)
            if num_regions > 1:
                original_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
                token_lengths = original_lengths * num_regions
                cu_seqlens = torch.zeros_like(cu_seqlens)
                cu_seqlens[1:] = torch.cumsum(token_lengths, dim=0)
                max_seqlen = int(max_seqlen * num_regions)
                if seq_idx is not None:
                    seq_idx = seq_idx.repeat_interleave(num_regions)
                if physical_positions is not None:
                    physical_positions = physical_positions.repeat_interleave(num_regions, dim=0)
        
        # Physical positional encoding (packed: positions is 1-D [total_seq_len])
        if self.physical_pe and physical_positions is not None:
            logits = self._apply_physical_pe(logits, physical_positions)

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
            raw_x = self._flatten_raw_tiled_regions(raw_x, packed=True)
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
        seq_lengths=None,
        physical_positions=None,
        fm_ids=None,
        **_,
    ):
        """Forward pass main function."""
        self._last_fm_stats = None
        # Prepare slice features [B, num_slices, embed_dim]:
        #   - inference: FM set (K may be 1), fused across FMs
        #   - subset SSL: FM set per view, fused across FMs
        #   - pair SSL: single-FM view, no FM fusion
        if self.mode == "inference":
            logits, token_lengths, num_regions = self._prepare_inference_fm_set(
                x, seq_lengths=seq_lengths, fm_ids=fm_ids
            )
        elif fm_ids is not None:
            logits, token_lengths, num_regions = self._prepare_subset_fm_set(
                x, input_feature_dims, fm_ids=fm_ids, seq_lengths=seq_lengths
            )
        else:
            logits, token_lengths, num_regions = self._prepare_pair_ssl_view(
                x, input_feature_dims, seq_lengths=seq_lengths
            )

        # Physical positional encoding (padded: positions is [B, num_tokens]).
        if self.physical_pe and physical_positions is not None:
            token_positions = self._expand_physical_positions_for_regions(
                physical_positions, num_regions
            )
            logits = self._apply_physical_pe(logits, token_positions, token_lengths)

        # Build attention mask if sequence lengths are provided.
        mask = None
        if token_lengths is not None:
            max_len = logits.shape[1]  
            mask = self._build_mask(token_lengths, max_len)  # [B, num_tokens]

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

        # Return token-level embeddings before aggregation
        if return_slice_embeddings:
            if self.slice_pooling == "cls":
                return h[:, 1:, :]  # Remove CLS token
            return h  # [B, num_tokens, embed_dim]

        # Slice feature aggregation
        # CLS token pooling
        if self.slice_pooling == "cls":
            if get_attention or get_per_head_attention:
                return self._extract_cls_attention(logits, mask, src_key_padding_mask)
            pooled = h[:, 0, :]
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
            raw_x = self._flatten_raw_tiled_regions(raw_x, packed=False)
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
        use_packed: bool = False,
        physical_positions=None,
        fm_ids=None,
        **kwargs,
    ):
        """
        Forward pass through the Cobra network.
        Input data cases:
            Pair SSL (single-FM view), padded:
                x = Tensor [B, num_slices, input_embed_dim]
            Pair SSL (single-FM view), packed (use_packed=True):
                x = Tensor [total_slices, input_embed_dim]
            Subset SSL (FM set per view), padded:
                x = Tensor [B, K, num_slices, input_embed_dim] with fm_ids [B, K]
            Inference, single FM:
                x = Tensor [B, num_slices, input_embed_dim] (normalized to a list of 1),
                or List with a single tensor of that shape.
            Inference, FM set (K >= 1):
                x = List of K tensors, each [B, num_slices, input_embed_dim]

        All FM-set paths (single-/multi-FM inference and subset SSL) normalize to [K, B, num_slices, embed_dim] before FM fusion, while pair/packed SSL stay
        as single-FM views.

        Args:
            x: Input tensor or list of tensors (see public input cases above).
            input_feature_dims: Feature dimensions per sample [B] (SSL mode).
            get_attention: If True, return aggregated attention map [B, 1, num_tokens].
            get_per_head_attention: If True, return per-head attention [B, num_heads, num_tokens] (ABMIL only).
            return_slice_embeddings: If True, return token-level embeddings [B, num_tokens, embed_dim] before pooling.
            use_packed: If True, use packed sequence for variable sequence length handling.
            physical_positions: Normalized relative slice depth in [0, 1] for sinusoidal PE.
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
            If get_attention=True: Attention map [B, 1, num_tokens]
            If get_per_head_attention=True: Per-head attention [B, num_heads, num_tokens]
            If return_slice_embeddings=True: Token embeddings [B, num_tokens, embed_dim]
            Otherwise: Features [B, contrast_dim] (train) or [B, output_dim] (inference)
        """
        # Single-FM inference may be passed as a bare tensor; treat it as an FM set of size 1
        # so single- and multi-FM inference share the same FM-set path.
        if self.mode == "inference" and isinstance(x, torch.Tensor):
            x = [x]

        if use_packed:
            if self.mode == "inference":
                raise NotImplementedError(
                    "Packed inference is not supported. Use padded inputs "
                    "(list of [B, num_slices, F] tensors plus seq_lengths)."
                )
            if fm_ids is not None:
                raise NotImplementedError(
                    "Packed subset FM mode is not supported yet. Set use_packed=False for multi-FM subset SSL."
                )
            if kwargs.get('cu_seqlens') is None or kwargs.get('max_seqlen') is None or kwargs.get('seq_idx') is None:
                raise ValueError("cu_seqlens, max_seqlen, and seq_idx are required for packed mode.")
            return self._forward_packed(
                x,
                input_feature_dims=input_feature_dims,
                get_attention=get_attention,
                return_slice_embeddings=return_slice_embeddings,
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
                physical_positions=physical_positions,
                fm_ids=fm_ids,
                **kwargs,
            )