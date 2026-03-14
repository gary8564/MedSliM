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
[3] Khosla, Prannay, et al. "Supervised Contrastive Learning." NeurIPS 2020.
    https://arxiv.org/abs/2004.11362
[4] Tian, Yonglong, et al. "StableRep: Synthetic Images from Text-to-Image Models
    Make Strong Visual Representation Learners." CVPR 2024.
    https://arxiv.org/abs/2306.00984
[5] Guinot, Julien, et al. "Semi-Supervised Contrastive Learning of Musical
    Representations." ISMIR 2024. https://arxiv.org/abs/2407.13840
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from accelerate import Accelerator

from med_slim.model.sequence_encoder.cobra import Cobra

class MoCo(nn.Module): 
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs.
    
    Supports semi-supervised contrastive learning: when classification labels are
    available for a subset of samples, samples sharing labels are treated as
    additional positives (SupCon), while unlabeled samples use standard InfoNCE.
    This follows the SemiSupCon framework [5] using the StableRep multi-positive
    contrastive loss formulation [4].
    
    References:
        MoCo v3 (Chen et al., 2021) - https://arxiv.org/abs/2104.02057
    """

    def __init__(
        self,
        embed_dim: int,
        contrast_dim: int,
        accelerator: Optional[Accelerator] = None,
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
                         If None, single-process mode (no gathering across GPUs).
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

    def _gather(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather tensor across all processes. No-op when accelerator is None (single process)."""
        if self.accelerator is not None and self.accelerator.num_processes > 1:
            return self.accelerator.gather(tensor)
        return tensor

    @property
    def _rank(self) -> int:
        """Current process rank. Returns 0 for single-process."""
        if self.accelerator is not None:
            return self.accelerator.process_index
        return 0

    @torch.no_grad()
    def _update_momentum_encoder(self, m=0.99):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def _build_label_positive_mask(
        self,
        labels: torch.Tensor,
        has_label: torch.Tensor,
        all_labels: torch.Tensor,
        all_has_label: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build a label-based positive mask for multi-label supervised contrastive learning.
        
        Two samples are label-positives if:
        - Both are labeled, AND
        - They share at least one common positive label (unhealthy), OR both are all-negative (healthy).
        
        Args:
            labels: Local query labels [batch_size, C] (binary multi-label vectors).
            has_label: Local query label availability [batch_size] (boolean).
            all_labels: Gathered labels from all processes [batch_size * world_size, C].
            all_has_label: Gathered label availability [batch_size * world_size] (boolean).
            
        Returns:
            Label-based positive mask [batch_size, batch_size * world_size] (float, 0/1 values).
        """
        # Mask for pairs where both query and key are labeled
        both_labeled = has_label.float().unsqueeze(1) * all_has_label.float().unsqueeze(0)  # [batch_size, batch_size * world_size]
        
        # Shared positive labels: dot product of binary label (overlapping positives)
        shared_positives = torch.mm(labels.float(), all_labels.float().T)  # [batch_size, batch_size * world_size]
        
        # Both-negative (healthy): both samples have no positive labels
        is_neg_q = (labels.sum(dim=1) == 0).float()       # [batch_size]
        is_neg_k = (all_labels.sum(dim=1) == 0).float()   # [batch_size * world_size]
        both_negative = torch.outer(is_neg_q, is_neg_k)   # [batch_size, batch_size * world_size]
        
        # Label-based positives: (shared positive labels OR both healthy) AND both labeled
        label_pos = ((shared_positives > 0).float() + both_negative).clamp(max=1.0)
        label_pos = label_pos * both_labeled
        
        return label_pos

    def contrastive_loss(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        has_label: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute contrastive loss with optional label-based multi-positive supervision
        which unifies InfoNCE and SupCon in a single loss.
        - Unlabeled samples: InfoNCE.
        - Labeled samples: SupCon.
        
        Args:
            q: Query embeddings [N, D] from base encoder + predictor.
            k: Key embeddings [N, D] from momentum encoder (before gathering).
            labels: Optional multi-label tensor [N, C].
                    Only used for samples where has_label=True. Ignored when None.
            has_label: Optional boolean tensor [N] indicating which samples have labels.
            
        Returns:
            Scalar contrastive loss.
        """
        # Normalize embeddings
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        
        # Gather all keys across GPUs
        with torch.no_grad():
            k = self._gather(k)  # [world_size * batch_size, contrast_dim]
        
        # Compute similarity logits
        logits = torch.einsum('nc, mc -> nm', [q, k]) / self.T  # # Einstein sum [batch_size, world_size * batch_size]
        N = logits.shape[0] # batch size per GPU
        rank = self._rank
        
        # Determine whether to use semi-supervised loss
        use_supcon = (labels is not None and has_label is not None and has_label.any())
        
        if not use_supcon:
            # Standard InfoNCE: one-hot target at the cross-model view position
            targets = torch.arange(N, dtype=torch.long, device=logits.device) + N * rank  # for query i, the correct class index is i (or i + offset) (N * rank is for multi-GPU setting)
            return nn.CrossEntropyLoss()(logits, targets) * (2 * self.T)
        
        # Semi-supervised contrastive loss
        N_global = logits.shape[1] # world_size * batch_size
        
        # 1. Cross-view positives (InfoNCE): each query's augmented view is always a positive
        pos_mask = torch.zeros(N, N_global, device=logits.device)
        local_indices = torch.arange(N, device=logits.device)
        pos_mask[local_indices, local_indices + N * rank] = 1.0
        
        # 2. Label-based positives (SupCon): same-class keys are also positives
        with torch.no_grad():
            all_labels = self._gather(labels.clone())       # [N_global, C]
            all_has_label = self._gather(has_label.clone())  # [N_global]
        
        label_pos = self._build_label_positive_mask(labels, has_label, all_labels, all_has_label)
        
        # 3. Union of cross-view and label-based positives
        pos_mask = (pos_mask + label_pos).clamp(max=1.0)
        
        # Create soft target distribution (normalized positive mask)
        # For unlabeled queries: one-hot → degenerates to InfoNCE
        # For labeled queries: uniform over all positives → SupCon
        p = pos_mask / pos_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        
        # Cross-entropy with soft targets: -sum(p * log_softmax(logits))
        # F.log_softmax can handle numerical stability (max subtraction) internally
        log_prob = F.log_softmax(logits, dim=1)
        loss = -(p * log_prob).sum(dim=1).mean()
        
        return loss * (2 * self.T)

    def forward(
        self, 
        x1, 
        x2, 
        *,
        input_feature_dims_1: torch.Tensor | None = None, 
        input_feature_dims_2: torch.Tensor | None = None, 
        m: float = 0.99,
        seq_lengths: torch.Tensor = None,
        labels: torch.Tensor | None = None,
        has_label: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """        
        Args:
            x1: First view features [B, num_slices, feature_dim]
            x2: Second view features [B, num_slices, feature_dim]
            input_feature_dims_1: Original feature dims per-sample for x1 with shape [B].
            input_feature_dims_2: Original feature dims per-sample for x2 with shape [B].
            m: Momentum parameter for momentum encoder update. Default: 0.99.
            seq_lengths: Actual sequence lengths with shape [B] for masking padded positions.
                         Shared across both views since positive pairs come from the same exam/plane.
            labels: Optional multi-label tensor [B, C] for semi-supervised contrastive learning.
                    Each row is a binary vector indicating class labels [abnormal, acl, meniscus].
                    Only used for samples where has_label=True. When None, InfoNCE is used.
            has_label: Optional boolean tensor [B] indicating which samples have classification labels.
                       Unlabeled samples use standard InfoNCE; labeled samples use SupCon multi-positive targets.
            
        Returns:
            Contrastive loss.
        """
        # Compute query features
        q1 = self.predictor(self.base_encoder(
            x1, input_feature_dims=input_feature_dims_1, seq_lengths=seq_lengths
        ))
        q2 = self.predictor(self.base_encoder(
            x2, input_feature_dims=input_feature_dims_2, seq_lengths=seq_lengths
        ))
       
        with torch.no_grad():
            self._update_momentum_encoder(m=m) # Update the momentum encoder
            # Compute key features through momentum encoder
            k1 = self.momentum_encoder(
                x1, input_feature_dims=input_feature_dims_1, seq_lengths=seq_lengths
            )
            k2 = self.momentum_encoder(
                x2, input_feature_dims=input_feature_dims_2, seq_lengths=seq_lengths
            )

        return (
            self.contrastive_loss(q1, k2, labels=labels, has_label=has_label) + 
            self.contrastive_loss(q2, k1, labels=labels, has_label=has_label)
        )