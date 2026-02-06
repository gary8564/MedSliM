"""
Adapted from:
[1] https://github.com/KatherLab/COBRA/blob/main/cobra/utils/mamba2.py
Lenz, Tim, Peter Neidlinger, Marta Ligero, Georg Wölflein, Marko van Treeck and Jakob Nikolas Kather. 
Unsupervised Foundation Model-Agnostic Slide-Level Representation Learning.
2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR): 30807-30817, 2024.
[2] https://github.com/facebookresearch/moco-v3
Chen, Xinlei, Saining Xie and Kaiming He.
An Empirical Study of Training Self-Supervised Vision Transformers.
2021 IEEE/CVF International Conference on Computer Vision (ICCV) (2021): 9620-9629.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from accelerate import Accelerator

from med_slim.model.sequence_encoder.cobra import Cobra

class MoCo(nn.Module): 
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """

    def __init__(
        self,
        embed_dim: int,
        contrast_dim: int,
        accelerator: Accelerator,
        input_dims: Optional[List[int]] = None,
        num_heads: int = 8,
        num_layers: int = 2,
        T: float = 0.2,
        dropout: float = 0.25,
        sequence_encoder: str = "mamba2",
        pooling: str = "abmil",
        **kwargs,
    ):
        """
        Args:
            embed_dim: Internal embedding dimensionality.
            contrast_dim: Output dimensionality for contrastive features.
            accelerator: HuggingFace Accelerator for distributed training.
            input_dims: List of input feature dimensions to support.
            num_heads: Number of attention heads.
            num_layers: Number of layers in the sequence encoder.
            T: Softmax temperature for contrastive loss.
            dropout: Dropout rate.
            sequence_encoder: "mamba2" (default) or "transformer".
            pooling: Slice pooling method - "abmil" (default) or "cls" (requires transformer).
            **kwargs: Encoder/pooling-specific parameters (passed to Cobra):
                - d_state (int): Mamba2 internal state dim (default: 128)
                - dim_feedforward (int): Transformer FFN hidden size (default: 4 * embed_dim)
                - norm_first (bool): Pre-LN transformer (default: True)
                - rotary_positional_encoding (str): "RoPE" or None (default: None)
                - att_dim (int): ABMIL attention hidden dim (default: 256)
        """
        super().__init__()

        if input_dims is None:
            input_dims = [512, 768, 1024, 1152, 1376, 1536]

        self.T = T
        self.accelerator = accelerator

        # Shared encoder kwargs
        encoder_kwargs = dict(
            embed_dim=embed_dim,
            contrast_dim=contrast_dim,
            input_dims=input_dims,
            num_heads=num_heads,
            num_layers=num_layers,
            sequence_encoder=sequence_encoder,
            slice_pooling=pooling,
            **kwargs,
        )

        self.base_encoder = Cobra(dropout=dropout, **encoder_kwargs)
        self.momentum_encoder = Cobra(dropout=0.0, **encoder_kwargs)  # No dropout for momentum encoder
        self.predictor = nn.Sequential(
            nn.LayerNorm(contrast_dim),
            nn.Linear(contrast_dim,2 * contrast_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(2 * contrast_dim,contrast_dim),
            nn.BatchNorm1d(contrast_dim),
        )

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)  # initialize the momentum encoder with the base encoder
            param_m.requires_grad = False # No gradient updates for the momentum encoder

    @torch.no_grad()
    def _update_momentum_encoder(self, m=0.99):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def contrastive_loss(self, q, k):
        """Compute InfoNCE contrastive loss."""
        # normalize
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        
        # gather all targets
        with torch.no_grad():
            k = self.accelerator.gather(k)  # [world_size * batch_size, contrast_dim]
        
        # Einstein sum is more intuitive
        logits = torch.einsum('nc, mc -> nm', [q, k]) / self.T # shape [batch_size, world_size * batch_size]
        N = logits.shape[0] # batch size per GPU
        rank = self.accelerator.process_index
        labels = torch.arange(N, dtype=torch.long, device=logits.device) + N * rank # for query i, the correct class index is i (or i + offset) (N * rank is for multi-GPU setting)
        
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

    def _forward_padded(
        self,
        x1,
        x2,
        input_feature_dims_1: torch.Tensor = None,
        input_feature_dims_2: torch.Tensor = None,
        m: float = 0.99,
        seq_lengths_1: torch.Tensor = None,
        seq_lengths_2: torch.Tensor = None,
        **_, 
    ):
        """Forward pass for padded sequences."""
        # Compute the contrastive features 
        q1 = self.predictor(self.base_encoder(
            x1, input_feature_dims=input_feature_dims_1, seq_lengths=seq_lengths_1
        ))
        q2 = self.predictor(self.base_encoder(
            x2, input_feature_dims=input_feature_dims_2, seq_lengths=seq_lengths_2
        ))
       
        with torch.no_grad():
            self._update_momentum_encoder(m=m) # Update the momentum encoder
            # Compute the contrastive features for the momentum encoder as targets
            k1 = self.momentum_encoder(
                x1, input_feature_dims=input_feature_dims_1, seq_lengths=seq_lengths_1
            )
            k2 = self.momentum_encoder(
                x2, input_feature_dims=input_feature_dims_2, seq_lengths=seq_lengths_2
            )

        return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)
    
    def _forward_packed(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        input_feature_dims_1: torch.Tensor = None,
        input_feature_dims_2: torch.Tensor = None,
        m: float = 0.99,
        cu_seqlens1: torch.Tensor = None,
        cu_seqlens2: torch.Tensor = None,
        max_seqlen1: int = None,
        max_seqlen2: int = None,
        seq_idx1: torch.Tensor = None,
        seq_idx2: torch.Tensor = None,
        **_,  # Ignore extra kwargs
    ):
        """Forward pass for packed sequences (no padding waste)."""
        # Compute the contrastive features 
        q1 = self.predictor(self.base_encoder(
            x1, 
            input_feature_dims=input_feature_dims_1,
            use_packed=True,
            cu_seqlens=cu_seqlens1,
            max_seqlen=max_seqlen1,
            seq_idx=seq_idx1,
        ))
        q2 = self.predictor(self.base_encoder(
            x2,
            input_feature_dims=input_feature_dims_2,
            use_packed=True,
            cu_seqlens=cu_seqlens2,
            max_seqlen=max_seqlen2,
            seq_idx=seq_idx2,
        ))
       
        with torch.no_grad():
            self._update_momentum_encoder(m=m)
            k1 = self.momentum_encoder(
                x1,
                input_feature_dims=input_feature_dims_1,
                use_packed=True,
                cu_seqlens=cu_seqlens1,
                max_seqlen=max_seqlen1,
                seq_idx=seq_idx1,
            )
            k2 = self.momentum_encoder(
                x2,
                input_feature_dims=input_feature_dims_2,
                use_packed=True,
                cu_seqlens=cu_seqlens2,
                max_seqlen=max_seqlen2,
                seq_idx=seq_idx2,
            )

        return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)

    def forward(
        self, 
        x1, 
        x2, 
        *,
        input_feature_dims_1: torch.Tensor | None = None, 
        input_feature_dims_2: torch.Tensor | None = None, 
        m: float = 0.99,
        use_packed: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """        
        Args:
            x1: First view features
                - Padded mode: [B, max_seq_len, feature_dim]
                - Packed mode: [total_seq_len, feature_dim]
            x2: Second view features with same shape as x1.
            input_feature_dims_1: Original feature dims per-sample for x1 with shape [B].
            input_feature_dims_2: Original feature dims per-sample for x2 with shape [B].
            m: Momentum parameter for momentum encoder update. Default: 0.99.
            use_packed: If True, use packed sequence for variable sequence length handling.
            
            **kwargs: Additional mode-specific parameters for variable sequence length handling:
                Padded mode:
                    - seq_lengths_1: Actual sequence lengths for x1 with shape [B]
                    - seq_lengths_2: Actual sequence lengths for x2 with shape [B]
                Packed mode:
                    - cu_seqlens1, cu_seqlens2: Cumulative sequence lengths with shape [B+1]
                    - max_seqlen1, max_seqlen2: Max sequence length in batch for x1 and x2
                    - seq_idx1, seq_idx2: Document index per token with shape [total_seq_len]
            
        Returns:
            Contrastive loss.
        """
        if use_packed:
            return self._forward_packed(
                x1, x2,
                input_feature_dims_1=input_feature_dims_1,
                input_feature_dims_2=input_feature_dims_2,
                m=m,
                **kwargs,
            )
        else:
            return self._forward_padded(
                x1, x2,
                input_feature_dims_1=input_feature_dims_1,
                input_feature_dims_2=input_feature_dims_2,
                m=m,
                **kwargs,
            )
    