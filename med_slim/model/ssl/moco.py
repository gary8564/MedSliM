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
[6] Assran, Mahmoud, et al. "Self-Supervised Learning from Images with a Joint-Embedding
    Predictive Architecture." CVPR 2023. https://arxiv.org/abs/2301.08243
[7] Mur-Labadia, Lorenzo, et al. "V-JEPA 2.1: Unlocking Dense Features in Video
    Self-Supervised Learning." 2026. https://arxiv.org/abs/2603.14482
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
from accelerate import Accelerator

from med_slim.model.sequence_encoder.cobra import Cobra
from med_slim.model.ssl.msp_predictor import MSPPredictor
from med_slim.model.ssl.masking import (
    generate_contiguous_slice_mask,
    generate_contiguous_slice_mask_packed,
)

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
        physical_pe: bool = False,
        regional_tokens: int = 0,
        region_embedding: bool = False,
        msp_enabled: bool = False,
        msp_lambda_mask: float = 1.0,
        msp_lambda_ctx: float = 0.0,
        msp_mask_ratio: tuple = (0.3, 0.5),
        msp_predictor_depth: int = 2,
        msp_predictor_dim: int | None = None,
        msp_max_seq_len: int = 512,
        msp_ctx_distance_weighted: bool = True,
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
            pooling: Slice pooling method - "abmil" (default), "cross_attention", or "cls" (requires transformer).
            physical_pe: Add sinusoidal positional encoding from physical
                slice positions (mm) before the sequence encoder.
            msp_enabled: Enable Masked Slice Prediction auxiliary objective.
            msp_lambda_mask: Weight for MSP loss on masked positions.
            msp_lambda_ctx: Base weight λ for context loss on visible positions (V-JEPA 2.1).  Set to 0 to disable.  
                When msp_ctx_distance_weighted=True, each visible slice receives
                per-token weight λ_i = λ / sqrt(d_min) following Eq. 3 of V-JEPA 2.1 (Mur-Labadia et al., 2026).
                When msp_ctx_distance_weighted=False, all visible slices contribute equally (uniform weight).
            msp_mask_ratio: (min_ratio, max_ratio) for contiguous masking.
            msp_predictor_depth: Number of residual MLP blocks in the MSP predictor.
            msp_predictor_dim: Hidden dimension of the MSP predictor. Defaults to embed_dim // 4.
            msp_max_seq_len: Maximum sequence length for positional embeddings.
            msp_ctx_distance_weighted: Use V-JEPA 2.1 inverse-sqrt-distance weighting for context loss (Eq. 3).  
                When msp_ctx_distance_weighted=False, all visible slices contribute equally (uniform weight).
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
        self.msp_enabled = msp_enabled
        self.msp_lambda_mask = msp_lambda_mask
        self.msp_lambda_ctx = msp_lambda_ctx
        self.msp_mask_ratio = tuple(msp_mask_ratio)
        self.msp_ctx_distance_weighted = msp_ctx_distance_weighted

        # Shared encoder kwargs
        encoder_kwargs = dict(
            embed_dim=embed_dim,
            contrast_dim=contrast_dim,
            input_dims=input_dims,
            num_heads=num_heads,
            num_layers=num_layers,
            sequence_encoder=sequence_encoder,
            slice_pooling=pooling,
            physical_pe=physical_pe,
            regional_tokens=regional_tokens,
            region_embedding=region_embedding,
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

        # MSP components
        if self.msp_enabled:
            self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
            nn.init.normal_(self.mask_token, std=0.02)
            self.msp_predictor = MSPPredictor(
                embed_dim=embed_dim,
                hidden_dim=msp_predictor_dim,
                num_layers=msp_predictor_depth,
                max_seq_len=msp_max_seq_len,
                dropout=dropout,
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

    @staticmethod
    def _distance_weights(
        visible_positions: torch.Tensor,
        mask_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute inverse-sqrt-distance weights for context tokens.

        Implements Eq. 3 of Mur-Labadia et al. (2026):
            λ_i = λ / sqrt(d_min(i, M))

        The base λ is applied externally via msp_lambda_ctx.

        Args:
            visible_positions: 1-D indices of visible slices [N_vis].
            mask_positions: 1-D indices of masked slices [N_mask].

        Returns:
            Normalized per-token weights [N_vis] (float, same device).
        """
        dists = (visible_positions.unsqueeze(1) - mask_positions.unsqueeze(0)).abs()  # [N_vis, N_mask]
        d_min = dists.min(dim=1).values.float().clamp(min=1.0)  # [N_vis]
        w = 1.0 / d_min.sqrt()
        w = w * (w.numel() / w.sum().clamp(min=1e-8))
        return w

    def _compute_msp_loss(
        self,
        x1: torch.Tensor,
        *,
        input_feature_dims_1: torch.Tensor | None = None,
        seq_lengths: torch.Tensor = None,
        physical_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Masked Slice Prediction loss.

        Masks a contiguous block of slices from one FM-view, runs the student
        encoder on the masked sequence, and predicts the teacher encoder's
        hidden states at masked positions via the MSP predictor.

        Returns:
            (loss_msp, loss_ctx): MSP loss on masked positions and optional
            context loss on visible positions.
        """
        B, num_slices, _ = x1.shape
        device = x1.device

        # Determine valid lengths per sample
        if seq_lengths is not None:
            eff_lengths = seq_lengths
        else:
            eff_lengths = torch.full((B,), num_slices, dtype=torch.long, device=device)

        # Generate contiguous slice mask [B, num_slices]
        slice_mask = generate_contiguous_slice_mask(
            eff_lengths, mask_ratio_range=self.msp_mask_ratio, max_seq_len=num_slices,
        ).to(device)

        # Student: mask and encode masked FM-view to get per-slice hidden states
        h_student = self.base_encoder(
            x1,
            input_feature_dims=input_feature_dims_1,
            seq_lengths=seq_lengths,
            return_slice_embeddings=True,
            slice_mask=slice_mask,
            mask_token=self.mask_token,
            physical_positions=physical_positions,
        )  # [B, num_slices, embed_dim]

        # Teacher: encode full FM-view to get per-slice hidden states (targets)
        with torch.no_grad():
            h_teacher = self.momentum_encoder(
                x1,
                input_feature_dims=input_feature_dims_1,
                seq_lengths=seq_lengths,
                return_slice_embeddings=True,
                physical_positions=physical_positions,
            )  # [B, num_slices, embed_dim]

        # Build valid mask (exclude padding from loss)
        valid_mask = torch.ones(B, num_slices, dtype=torch.bool, device=device)
        if seq_lengths is not None:
            positions = torch.arange(num_slices, device=device).unsqueeze(0)
            valid_mask = positions < seq_lengths.unsqueeze(1)

        # MSP loss: masked slice positions only
        masked_valid = slice_mask & valid_mask  # [B, num_slices]
        h_s_masked = h_student[masked_valid]    # [N_masked, embed_dim]
        h_t_masked = h_teacher[masked_valid]    # [N_masked, embed_dim]

        # Position indices (column index within each sample)
        pos_masked = masked_valid.nonzero()[:, 1]  # [N_masked]

        h_pred = self.msp_predictor(h_s_masked, pos_masked)
        loss_msp = F.mse_loss(h_pred, h_t_masked.detach())

        # Context loss: visible (non-masked and non-padded) slice positions
        loss_ctx = torch.tensor(0.0, device=device)
        if self.msp_lambda_ctx > 0:
            visible_valid = (~slice_mask) & valid_mask
            if visible_valid.any():
                h_s_vis = h_student[visible_valid]     # [N_vis, embed_dim]
                h_t_vis = h_teacher[visible_valid]

                if self.msp_ctx_distance_weighted:
                    vis_pos = visible_valid.nonzero()[:, 1]
                    mask_pos = masked_valid.nonzero()[:, 1]
                    w = self._distance_weights(vis_pos, mask_pos)  # [N_vis]
                    per_token = ((h_s_vis - h_t_vis.detach()) ** 2).mean(dim=1)
                    loss_ctx = (w * per_token).mean()
                else:
                    loss_ctx = F.mse_loss(h_s_vis, h_t_vis.detach())

        return loss_msp, loss_ctx

    def _compute_msp_loss_packed(
        self,
        x1: torch.Tensor,
        *,
        input_feature_dims_1: torch.Tensor = None,
        cu_seqlens1: torch.Tensor = None,
        max_seqlen1: int = None,
        seq_idx1: torch.Tensor = None,
        physical_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Masked Slice Prediction loss (packed mode).

        Same logic as _compute_msp_loss but operates on packed tensors.

        Returns:
            (loss_msp, loss_ctx)
        """
        device = x1.device

        # Generate contiguous slice mask [total_seq_len]
        slice_mask = generate_contiguous_slice_mask_packed(
            cu_seqlens1, mask_ratio_range=self.msp_mask_ratio,
        ).to(device)

        # Student: encode masked view → per-slice hidden states
        h_student = self.base_encoder(
            x1,
            input_feature_dims=input_feature_dims_1,
            use_packed=True,
            cu_seqlens=cu_seqlens1,
            max_seqlen=max_seqlen1,
            seq_idx=seq_idx1,
            return_slice_embeddings=True,
            slice_mask=slice_mask,
            mask_token=self.mask_token,
            physical_positions=physical_positions,
        )  # [total_seq_len, embed_dim]

        # Teacher: encode full view → per-slice hidden states
        with torch.no_grad():
            h_teacher = self.momentum_encoder(
                x1,
                input_feature_dims=input_feature_dims_1,
                use_packed=True,
                cu_seqlens=cu_seqlens1,
                max_seqlen=max_seqlen1,
                seq_idx=seq_idx1,
                return_slice_embeddings=True,
                physical_positions=physical_positions,
            )  # [total_seq_len, embed_dim]

        # MSP loss: masked slice positions only
        h_s_masked = h_student[slice_mask]  # [N_masked, embed_dim]
        h_t_masked = h_teacher[slice_mask]

        # Local positions within each sequence
        global_indices = slice_mask.nonzero().squeeze(-1)
        sample_ids = seq_idx1[global_indices].long()
        local_positions = global_indices - cu_seqlens1[sample_ids]

        h_pred = self.msp_predictor(h_s_masked, local_positions)
        loss_msp = F.mse_loss(h_pred, h_t_masked.detach())

        # Context loss: visible (non-masked and non-padded) slice positions
        loss_ctx = torch.tensor(0.0, device=device)
        if self.msp_lambda_ctx > 0:
            visible_mask = ~slice_mask
            if visible_mask.any():
                h_s_vis = h_student[visible_mask]
                h_t_vis = h_teacher[visible_mask]

                if self.msp_ctx_distance_weighted:
                    vis_global = visible_mask.nonzero().squeeze(-1)
                    w = self._distance_weights(vis_global, global_indices)
                    per_token = ((h_s_vis - h_t_vis.detach()) ** 2).mean(dim=1)
                    loss_ctx = (w * per_token).mean()
                else:
                    loss_ctx = F.mse_loss(h_s_vis, h_t_vis.detach())

        return loss_msp, loss_ctx

    def _forward(
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
        physical_positions: torch.Tensor | None = None,
    ) -> torch.Tensor | dict:
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
            has_label: Optional boolean tensor [B] indicating which samples have classification labels.
            physical_positions: Physical slice positions in mm [B, num_slices] for sinusoidal PE.
            
        Returns:
            Contrastive loss.
        """
        pe_kwargs = dict(physical_positions=physical_positions)

        q1 = self.predictor(self.base_encoder(
            x1, input_feature_dims=input_feature_dims_1, seq_lengths=seq_lengths,
            **pe_kwargs,
        ))
        q2 = self.predictor(self.base_encoder(
            x2, input_feature_dims=input_feature_dims_2, seq_lengths=seq_lengths,
            **pe_kwargs,
        ))
       
        with torch.no_grad():
            self._update_momentum_encoder(m=m)
            k1 = self.momentum_encoder(
                x1, input_feature_dims=input_feature_dims_1, seq_lengths=seq_lengths,
                **pe_kwargs,
            )
            k2 = self.momentum_encoder(
                x2, input_feature_dims=input_feature_dims_2, seq_lengths=seq_lengths,
                **pe_kwargs,
            )

        loss_infonce = (
            self.contrastive_loss(q1, k2, labels=labels, has_label=has_label) + 
            self.contrastive_loss(q2, k1, labels=labels, has_label=has_label)
        )

        if not self.msp_enabled:
            return loss_infonce

        # MSP branch: masked slice prediction on the first FM-view
        loss_msp, loss_ctx = self._compute_msp_loss(
            x1,
            input_feature_dims_1=input_feature_dims_1,
            seq_lengths=seq_lengths,
            physical_positions=physical_positions,
        )

        total_loss = (
            loss_infonce
            + self.msp_lambda_mask * loss_msp
            + self.msp_lambda_ctx * loss_ctx
        )

        return {
            "loss": total_loss,
            "loss_infonce": loss_infonce.detach(),
            "loss_msp": loss_msp.detach(),
            "loss_ctx": loss_ctx.detach(),
        }
    
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
        labels: torch.Tensor | None = None,
        has_label: torch.Tensor | None = None,
        physical_positions: torch.Tensor | None = None,
        **_,  # Ignore extra kwargs
    ):
        """Forward pass for packed sequences (no padding waste)."""
        packed_kwargs1 = dict(
            use_packed=True,
            cu_seqlens=cu_seqlens1,
            max_seqlen=max_seqlen1,
            seq_idx=seq_idx1,
            physical_positions=physical_positions,
        )
        packed_kwargs2 = dict(
            use_packed=True,
            cu_seqlens=cu_seqlens2,
            max_seqlen=max_seqlen2,
            seq_idx=seq_idx2,
            physical_positions=physical_positions,
        )

        q1 = self.predictor(self.base_encoder(
            x1, input_feature_dims=input_feature_dims_1, **packed_kwargs1,
        ))
        q2 = self.predictor(self.base_encoder(
            x2, input_feature_dims=input_feature_dims_2, **packed_kwargs2,
        ))
       
        with torch.no_grad():
            self._update_momentum_encoder(m=m)
            k1 = self.momentum_encoder(
                x1, input_feature_dims=input_feature_dims_1, **packed_kwargs1,
            )
            k2 = self.momentum_encoder(
                x2, input_feature_dims=input_feature_dims_2, **packed_kwargs2,
            )

        loss_infonce = (
            self.contrastive_loss(q1, k2, labels=labels, has_label=has_label) + 
            self.contrastive_loss(q2, k1, labels=labels, has_label=has_label)
        )

        if not self.msp_enabled:
            return loss_infonce

        # MSP branch: masked slice prediction on view_1
        loss_msp, loss_ctx = self._compute_msp_loss_packed(
            x1,
            input_feature_dims_1=input_feature_dims_1,
            cu_seqlens1=cu_seqlens1,
            max_seqlen1=max_seqlen1,
            seq_idx1=seq_idx1,
            physical_positions=physical_positions,
        )

        total_loss = (
            loss_infonce
            + self.msp_lambda_mask * loss_msp
            + self.msp_lambda_ctx * loss_ctx
        )

        return {
            "loss": total_loss,
            "loss_infonce": loss_infonce.detach(),
            "loss_msp": loss_msp.detach(),
            "loss_ctx": loss_ctx.detach(),
        }

    def forward(
        self, 
        x1, 
        x2, 
        *,
        input_feature_dims_1: torch.Tensor | None = None, 
        input_feature_dims_2: torch.Tensor | None = None, 
        m: float = 0.99,
        use_packed: bool = False,
        physical_positions: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | dict:
        """        
        Args:
            x1: First view features
                - Padded global-only mode: [B, max_seq_len, input_embed_dim]
                - Padded tiled mode: [B, max_seq_len, num_tiled_regions, input_embed_dim]
                - Packed global-only mode: [total_slices, input_embed_dim]
                - Packed tiled mode: [total_slices, num_tiled_regions, input_embed_dim]
            x2: Second view features with same shape as x1.
            input_feature_dims_1: Original feature dims per-sample for x1 with shape [B].
            input_feature_dims_2: Original feature dims per-sample for x2 with shape [B].
            m: Momentum parameter for momentum encoder update. Default: 0.99.
            use_packed: If True, use packed sequence for variable sequence length handling.
            physical_positions: Physical slice positions in mm for sinusoidal PE.
                - Padded mode: [B, num_slices]
                - Packed mode: [total_seq_len]
            
            **kwargs: Additional mode-specific parameters for variable sequence length handling:
                Padded mode:
                    - seq_lengths: Actual sequence lengths with shape [B]
                Packed mode:
                    - cu_seqlens1, cu_seqlens2: Cumulative sequence lengths with shape [B+1]
                    - max_seqlen1, max_seqlen2: Max sequence length in batch for x1 and x2
                    - seq_idx1, seq_idx2: Document index per token with shape [total_seq_len]
            
        Returns:
            When MSP is disabled: scalar contrastive loss.
            When MSP is enabled: dict with keys loss (total), loss_infonce, loss_msp, and loss_ctx.
        """
        if use_packed:
            return self._forward_packed(
                x1, x2,
                input_feature_dims_1=input_feature_dims_1,
                input_feature_dims_2=input_feature_dims_2,
                m=m,
                physical_positions=physical_positions,
                **kwargs,
            )
        else:
            return self._forward(
                x1, x2,
                input_feature_dims_1=input_feature_dims_1,
                input_feature_dims_2=input_feature_dims_2,
                m=m,
                physical_positions=physical_positions,
                **kwargs,
            )
    
