"""
Adapted from: https://github.com/KatherLab/COBRA/blob/main/cobra/model/model.py
Lenz, Tim, Peter Neidlinger, Marta Ligero, Georg Wölflein, Marko van Treeck and Jakob Nikolas Kather.
Unsupervised Foundation Model-Agnostic Slide-Level Representation Learning.
2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR): 30807-30817, 2024.
"""

from typing import List
from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


from .mamba2 import Mamba2Enc
from .transformer import TransformerEncoderLayer
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

        # Sequence encoder
        if self.sequence_encoder == "mamba2":
            self.seq_enc = Mamba2Enc(
                embed_dim,
                embed_dim,
                n_classes=embed_dim,
                layer=num_layers,
                dropout=dropout,
                d_state=kwargs.get('d_state', 128),
            )
        else:
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
            assert len(x)==len(input_feature_dims), f"Batch size mismatch between input x and input_feature_dims"
            logits = torch.concat([self.embed[str(input_feature_dims[i].item())](x[i,:,:input_feature_dims[i].item()]).unsqueeze(0) for i in range(len(x))], dim=0) # [B, num_slices, embed_dim]
        else:
            logits = self.embed[str(x.shape[-1])](x) # [B, num_slices, embed_dim]
        return logits

    def _embed_inference_forward(self, x) -> torch.Tensor:
        """Foundation model feature embedding in inference mode."""
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

    def forward(self, x, input_feature_dims=None, seq_lengths=None, get_attention=False):
        """
        Forward pass through the Cobra network.
        
        Args:
            x: Input tensor [B, num_slices, feature_dim] or list of tensors (inference mode).
            input_feature_dims: Feature dimensions per sample [B] (SSL mode).
            seq_lengths: Actual sequence lengths [B] for masking padded positions.
            get_attention: If True, return attention map instead of features.
        
        Returns:
            If get_attention=True: Attention map [B, 1, num_slices]
            Otherwise: Features [B, contrast_dim] (train) or [B, embed_dim] (inference)
        """
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

        # Slice feature aggregation
        # CLS token pooling
        if self.slice_pooling == "cls":
            if get_attention:
                return self._extract_cls_attention(logits, mask, src_key_padding_mask)
            pooled = h[:, 0, :]
            return self.proj(pooled) if self.mode == "train" else pooled

        # ABMIL pooling
        if self.num_heads > 1:
            # Split feature dim into heads: [B, num_slices, num_heads, head_dim]
            h_heads = rearrange(h, 'b t (e c) -> b t e c', c=self.num_heads)
            attentions = []
            for i, attn_net in enumerate(self.attn):
                _, raw_attention = attn_net(h_heads[:, :, :, i], mask=mask, return_raw_attention = True) # [B, num_slices, 1]
                attentions.append(raw_attention)
            A = torch.stack(attentions, dim=-1) # [B, num_slices, 1, num_heads]
            A = rearrange(A, 'b t e c -> b t (e c)',c=self.num_heads).mean(-1).unsqueeze(-1) # [B, num_slices, 1]
            A = torch.transpose(A, 2, 1) # [B, 1, num_slices]
            A = F.softmax(A, dim=-1) # [B, 1, num_slices]
        else:
            A = self.attn[0](h, mask=mask) 
            A = torch.transpose(A, 2, 1) # [B, 1, num_slices]

        if get_attention:
            return A

        # Training phase: MIL pooling over feature embedding after Mamba-encoder
        # Inference phase: MIL pooling over original input features
        if self.mode == "train":
            h = torch.bmm(A, h).squeeze(1) # [B, embed_dim]
            feats = self.proj(h)

        else:
            feats = torch.bmm(A, logits).squeeze(1)    # [B, embed_dim]

        assert len(feats.shape)==2, feats.shape
        return feats