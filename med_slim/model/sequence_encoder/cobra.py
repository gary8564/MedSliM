"""
Adapted from: https://github.com/KatherLab/COBRA/blob/main/cobra/model/model.py
Lenz, Tim, Peter Neidlinger, Marta Ligero, Georg Wölflein, Marko van Treeck and Jakob Nikolas Kather.
Unsupervised Foundation Model-Agnostic Slide-Level Representation Learning.
2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR): 30807-30817, 2024.
"""

from typing import List, Tuple
from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


from .mamba2 import Mamba2Enc
from .transformer import TransformerEncoderLayer, VarlenTransformerEncoder
from med_slim.model.attention_pooling import BatchedABMIL


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
        fm_pooling: str = "mean",
        slice_pooling: str = "abmil",
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
            fm_pooling: 'mean', 'concat', or 'attention' in inference mode.
                - 'mean': Average embeddings across foundation models.
                - 'concat': Concatenate FM embeddings along sequence dimension.
                - 'attention': Learn attention weights to pool FM embeddings per slice.
            slice_pooling: 'abmil' or 'cls' (cls requires transformer).
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
            assert fm_pooling in ["mean", "concat", "attention"], f"Invalid fm_pooling '{fm_pooling}'. Must be one of 'mean', 'concat', 'attention'."
        assert slice_pooling in ["abmil", "cls"], f"Invalid slice_pooling '{slice_pooling}'. Must be one of 'abmil', 'cls'."
        if slice_pooling == "cls" and sequence_encoder != "transformer":
                raise ValueError(f"slice_pooling='cls' requires sequence_encoder='transformer'. Got {sequence_encoder}.")

        self.mode = mode
        self.embed_dim = embed_dim
        self.sequence_encoder = sequence_encoder
        self.fm_pooling = fm_pooling
        self.slice_pooling = slice_pooling

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
                norm=nn.LayerNorm(embed_dim),
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

        # ABMIL pooling modules
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
        
        # FM attention pooling for inference mode
        # Learns to weight different foundation model embeddings at each slice position
        # TODO: fm_attn is randomly initialized in inference mode and not loaded from checkpoint.
        #       This is intended for FINE-TUNING COBRA (where fm_attn will be trained).
        #       For LINEAR PROBING (frozen COBRA), use fm_pooling='mean' or 'concat' instead.
        self.fm_attn = None
        if mode == "inference" and fm_pooling == "attention":
            self.fm_attn = BatchedABMIL(
                input_dim=embed_dim,
                hidden_dim=kwargs.get('att_dim', 256),
                dropout=0.5,
                n_heads=1,
                activation='softmax',
            )

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
        """Foundation model feature embedding in SSL pretraining mode."""
        if input_feature_dims is not None:
            assert len(x) == len(input_feature_dims), f"Batch size mismatch between input x and input_feature_dims"
            batch_size, seq_len, _ = x.shape
            
            # Group samples by their feature dimension to batch-process each group
            unique_dims = input_feature_dims.unique()
            
            if len(unique_dims) == 1:
                feat_dim = unique_dims[0].item()
                logits = self.embed[str(feat_dim)](x[:, :, :feat_dim])
            else:
                logits = None
                for dim in unique_dims:
                    feat_dim = dim.item()
                    mask = input_feature_dims == dim  # [batch_size]
                    x_group = x[mask, :, :feat_dim]  # [num_samples_with_dim, seq_len, feat_dim]
                    embedded = self.embed[str(feat_dim)](x_group)  # [num_samples_with_dim, seq_len, embed_dim]
                    if logits is None:
                        logits = torch.zeros(batch_size, seq_len, self.embed_dim, device=x.device, dtype=embedded.dtype)
                    logits[mask] = embedded
        else:
            logits = self.embed[str(x.shape[-1])](x)  # [B, num_slices, embed_dim]
        return logits

    def _embed_inference_forward(self, x) -> torch.Tensor:
        """Foundation model feature embedding in inference mode (zero padding)."""
        # Inference mode with ensembling feature embeddings from multiple slice encoders
        # x is a list of K tensors, each with shape [B, num_slices, encoder_embed_dim]
        embedded_features = [self.embed[str(xi.shape[-1])](xi) for xi in x]  # List of K [B, num_slices, embed_dim]
        fm_embs = torch.stack(embedded_features, dim=0)  # [K, B, num_slices, embed_dim]
        assert fm_embs.shape[-1] == self.embed_dim, f"Expected embed_dim {self.embed_dim}, got {fm_embs.shape[-1]}"
        assert len(fm_embs.shape)==4, f"Expected 4 dimensions, got {len(fm_embs.shape)}"
        assert fm_embs.shape[0]==len(x), f"Expected length of input x {len(x)}, got {fm_embs.shape[0]}"
        if fm_embs.shape[0] == 1:
            return fm_embs[0]
        if self.fm_pooling == "mean":
            # Average embeddings across different slice encoders
            logits = fm_embs.mean(dim=0)  # [B, num_slices, embed_dim]
        elif self.fm_pooling == "attention":
            # Attention-weighted pooling across different slice encoders
            # For each slice position, learn to weight the K FM embeddings
            K, B, T, E = fm_embs.shape
            # Reshape: [K, B, num_slices, embed_dim] -> [B*num_slices, K, embed_dim]
            fm_embs = rearrange(fm_embs, 'k b t e -> (b t) k e')
            # Apply attention: [B*num_slices, K, 1]
            attn_weights = self.fm_attn(fm_embs)  # [B*num_slices, K, 1]
            attn_weights_t = attn_weights.transpose(2, 1)  # [B*num_slices, 1, K]
            pooled = torch.bmm(attn_weights_t, fm_embs).squeeze(1)  # [B*num_slices, embed_dim]
            # Reshape back: [B*num_slices, embed_dim] -> [B, num_slices, embed_dim]
            logits = rearrange(pooled, '(b t) e -> b t e', b=B, t=T)
        else:
            # Concatenate embeddings across different slice encoders
            logits = rearrange(fm_embs, 'k b t e -> b (t k) e')  # [B, num_slices* K, embed_dim]
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
            x: Packed input tensor [total_seq_len, feature_dim]
            cu_seqlens: Cumulative sequence lengths [batch_size + 1], int32
            input_feature_dims: Feature dimensions per sample [batch_size] (optional)
        
        Returns:
            Embedded features [total_seq_len, embed_dim]
        """
        if input_feature_dims is not None:
            batch_size = cu_seqlens.shape[0] - 1
            total_seq_len = x.shape[0]
            
            # Calculate sequence lengths
            seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]  # [batch_size]
            
            # Group slices by their feature dimension for batch processing
            unique_dims = input_feature_dims.unique()
            
            if len(unique_dims) == 1:
                feat_dim = unique_dims[0].item()
                return self.embed[str(feat_dim)](x[:, :feat_dim])
            else:
                slice_feat_dims = torch.repeat_interleave(input_feature_dims, seq_lens)  # [total_seq_len]
                logits = None
                for dim in unique_dims:
                    feat_dim = dim.item()
                    mask = slice_feat_dims == dim  # [total_seq_len]
                    x_group = x[mask, :feat_dim]  # [num_slices_with_dim, feat_dim]
                    embedded = self.embed[str(feat_dim)](x_group)  # [num_slices_with_dim, embed_dim]
                    if logits is None:
                        logits = torch.zeros(total_seq_len, self.embed_dim, device=x.device, dtype=embedded.dtype)
                    logits[mask] = embedded
                return logits
        else:
            return self.embed[str(x.shape[-1])](x)  # [total_seq_len, embed_dim]

    def _embed_inference_forward_packed(self, x: List[torch.Tensor]) -> torch.Tensor:
        """
        Foundation model feature embedding in inference mode (packed sequences).
        
        Args:
            x: List of K tensors from different slice encoders, 
               each with shape [total_seq_len, encoder_embed_dim_k]
        
        Returns:
            Embedded features [total_seq_len, embed_dim]
        
        Raises:
            ValueError: 'concat' mode is not supported for packed sequences.
        """
        # Embed each encoder's features: List of K [total_seq_len, embed_dim]
        embedded_features = [self.embed[str(xi.shape[-1])](xi) for xi in x]
        
        fm_embs = torch.stack(embedded_features, dim=0)  # [K, total_seq_len, embed_dim]
        K, T, E = fm_embs.shape
        
        assert E == self.embed_dim, f"Expected embed_dim {self.embed_dim}, got {E}"
        
        if K == 1:
            return fm_embs[0]  # [total_seq_len, embed_dim]
        
        if self.fm_pooling == "mean":
            return fm_embs.mean(dim=0)  # [total_seq_len, embed_dim]
            
        elif self.fm_pooling == "attention":
            # Attention-weighted pooling across K encoders for each slice
            fm_embs_t = rearrange(fm_embs, 'k t e -> t k e')
            attn_weights = self.fm_attn(fm_embs_t)  # [total_seq_len, K, 1]
            attn_weights_t = attn_weights.transpose(2, 1)  # [total_seq_len, 1, K]
            return torch.bmm(attn_weights_t, fm_embs_t).squeeze(1)  # [total_seq_len, embed_dim]
            
        else:
            raise ValueError(f"fm_pooling='concat' is not supported for packed sequences.")

    def _packed_to_padded(
        self, 
        packed: torch.Tensor, 
        cu_seqlens: torch.Tensor, 
        max_seqlen: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert packed tensor to padded tensor with mask.
        
        Args:
            packed: Packed tensor [total_seq_len, dim]
            cu_seqlens: Cumulative sequence lengths [batch_size + 1]
            max_seqlen: Maximum sequence length
        
        Returns:
            padded: Padded tensor [batch_size, max_seqlen, dim]
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

    def _abmil_pooling(
        self, 
        h: torch.Tensor, 
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Apply ABMIL attention pooling.
        
        Args:
            h: Input tensor [batch_size, seq_len, embed_dim]
            mask: Boolean mask [batch_size, seq_len] (True = valid)
        
        Returns:
            Attention weights [batch_size, 1, seq_len]
        """
        if self.num_heads > 1:
            # Split feature dim into heads: [B, num_slices, num_heads, head_dim]
            h_heads = rearrange(h, 'b t (e c) -> b t e c', c=self.num_heads)
            attentions = []
            for i, attn_net in enumerate(self.attn):
                _, raw_attention = attn_net(h_heads[:, :, :, i], mask=mask, return_raw_attention=True) # [B, num_slices, 1]
                attentions.append(raw_attention)
            A = torch.stack(attentions, dim=-1) # [B, num_slices, 1, num_heads]
            A = rearrange(A, 'b t e c -> b t (e c)', c=self.num_heads).mean(-1).unsqueeze(-1) # [B, num_slices, 1]
            A = torch.transpose(A, 2, 1) # [B, 1, num_slices]
            A = F.softmax(A, dim=-1) # [B, 1, num_slices]
        else:
            A = self.attn[0](h, mask=mask) 
            A = torch.transpose(A, 2, 1) # [B, 1, num_slices]
        return A

    def _forward_padded(
        self, 
        x, 
        input_feature_dims=None, 
        get_attention=False, 
        return_slice_embeddings=False,
        seq_lengths=None,
        **_,  # Ignore extra kwargs
    ):
        """Forward pass for padded sequences."""
        # Foundation model feature embedding
        if self.mode == "inference":
            logits = self._embed_inference_forward(x)
        else:
            logits = self._embed_ssl_forward(x, input_feature_dims)

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
            if get_attention:
                return self._extract_cls_attention(logits, mask, src_key_padding_mask)
            pooled = h[:, 0, :]
            return self.proj(pooled) if self.mode == "train" else pooled

        # ABMIL pooling
        A = self._abmil_pooling(h, mask)

        if get_attention:
            return A

        # Training: MIL pooling over encoded features
        # Inference: MIL pooling over original input features
        if self.mode == "train":
            pooled = torch.bmm(A, h).squeeze(1)  # [B, embed_dim]
            return self.proj(pooled)
        else:
            return torch.bmm(A, logits).squeeze(1)  # [B, embed_dim]

    def _forward_packed(
        self, 
        x, 
        input_feature_dims: torch.Tensor = None,
        get_attention: bool = False,
        return_slice_embeddings: bool = False,
        cu_seqlens: torch.Tensor = None,
        max_seqlen: int = None,
        seq_idx: torch.Tensor = None,
        **_,  # Ignore extra kwargs
    ) -> torch.Tensor:
        """Forward pass for packed/variable-length sequences."""
        batch_size = cu_seqlens.shape[0] - 1
        
        # Foundation model feature embedding
        if self.mode == "inference":
            logits = self._embed_inference_forward_packed(x)
        else:
            logits = self._embed_ssl_forward_packed(x, cu_seqlens, input_feature_dims)
        
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
        # Convert to padded for batched ABMIL (efficient since ABMIL is cheap)
        h_padded, mask = self._packed_to_padded(h, cu_seqlens, max_seqlen)
        A = self._abmil_pooling(h_padded, mask)
        
        if get_attention:
            return [A[i, :, :cu_seqlens[i+1]-cu_seqlens[i]] for i in range(batch_size)]
        
        # Training: MIL pooling over feature embedding after Mamba-encoder
        # Inference: MIL pooling over original input features
        if self.mode == "train":
            pooled = torch.bmm(A, h_padded).squeeze(1)
            return self.proj(pooled)
        else:
            logits_padded, _ = self._packed_to_padded(logits, cu_seqlens, max_seqlen)
            return torch.bmm(A, logits_padded).squeeze(1)

    def forward(
        self, 
        x, 
        *,
        input_feature_dims=None, 
        get_attention=False, 
        return_slice_embeddings=False,
        use_packed: bool = False,
        **kwargs,
    ):
        """
        Forward pass through the Cobra network, supporting both padded and packed sequences.
        
        Args:
            x: Input tensor or list of tensors with shape of each
                - Padded mode: [B, num_slices, feature_dim]
                - Packed mode: [total_seq_len, feature_dim]
            input_feature_dims: Feature dimensions per sample [B] (SSL mode).
            get_attention: If True, return attention map instead of features.
            return_slice_embeddings: If True, return slice-level embeddings [B, num_slices, embed_dim] before pooling.
            use_packed: If True, use packed sequence for variable sequence length handling.
            
            **kwargs: Mode-specific parameters
                Padded mode:
                    - seq_lengths: Actual sequence lengths [B] for masking padded positions.
                Packed mode:
                    - cu_seqlens: Cumulative sequence lengths [B+1], int32
                    - max_seqlen: Maximum sequence length in the batch
                    - seq_idx: Document index for each token [total_seq_len], int32
        
        Returns:
            If get_attention=True: Attention map [B, 1, num_slices] or list of maps
            If return_slice_embeddings=True: Slice embeddings [B, num_slices, embed_dim] or packed
            Otherwise: Features [B, contrast_dim] (train) or [B, embed_dim] (inference)
        """
        if use_packed:
            return self._forward_packed(
                x,
                input_feature_dims=input_feature_dims,
                get_attention=get_attention,
                return_slice_embeddings=return_slice_embeddings,
                **kwargs,
            )
        else:
            return self._forward_padded(
                x,
                input_feature_dims=input_feature_dims,
                get_attention=get_attention,
                return_slice_embeddings=return_slice_embeddings,
                **kwargs,
            )