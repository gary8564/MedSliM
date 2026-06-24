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
        physical_pe: bool = False,
        regional_tokens: int = 0,
        fm_pooling: str = "avg_pool",
        num_fms: int | None = None,
        per_fm_adapter_mode: str = "per_dim",
        fm_input_dims: Optional[List[int]] = None,
        router_use_fm_embedding: bool = True,
        router_use_fm_logit_bias: bool = True,
        router_use_fm_logit_scale: bool = False,
        router_mode: str = "soft",
        router_top_k: int | None = None,
        router_temperature: float = 1.0,
        router_learnable_temperature: bool = False,
        router_load_balance_weight: float = 0.01,
        router_z_loss_weight: float = 0.0,
        router_entropy_weight: float = 0.0,
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
            physical_pe: Add sinusoidal positional encoding from normalized relative
                slice depth in [0, 1] before the sequence encoder.
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

        # FM fusion / router configuration
        if fm_pooling not in ("avg_pool", "router"):
            raise ValueError(
                f"Invalid fm_pooling '{fm_pooling}'. Must be one of 'avg_pool', 'router'."
            )
        self.fm_pooling = fm_pooling
        self.num_fms = num_fms
        self.router_load_balance_weight = router_load_balance_weight
        self.router_z_loss_weight = router_z_loss_weight
        self.router_entropy_weight = router_entropy_weight

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
            fm_pooling=fm_pooling,
            num_fms=num_fms,
            per_fm_adapter_mode=per_fm_adapter_mode,
            fm_input_dims=fm_input_dims,
            router_use_fm_embedding=router_use_fm_embedding,
            router_use_fm_logit_bias=router_use_fm_logit_bias,
            router_use_fm_logit_scale=router_use_fm_logit_scale,
            router_mode=router_mode,
            router_top_k=router_top_k,
            router_temperature=router_temperature,
            router_learnable_temperature=router_learnable_temperature,
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

    @staticmethod
    def aggregate_fm_usage(
        fm_weights_local: torch.Tensor,
        fm_ids: torch.Tensor,
        num_fms: int,
        slice_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Per-global-FM routing mass fraction for one forward stats dict.

        Aggregates slice-averaged local softmax weights into global FM IDs.
        Returns shape [num_fms] with non-negative entries that sum to 1 when
        any routing mass is present.
        """
        if slice_mask is not None:
            slice_valid = slice_mask.to(device=fm_weights_local.device, dtype=fm_weights_local.dtype)
            denom = slice_valid.sum(dim=1).clamp_min(1.0)
            importance = (fm_weights_local * slice_valid.unsqueeze(-1)).sum(dim=1) / denom[:, None]
        else:
            importance = fm_weights_local.mean(dim=1)  # [B, K]

        flat_ids = fm_ids.to(device=importance.device, dtype=torch.long).reshape(-1)
        flat_importance = importance.reshape(-1)
        usage = torch.zeros(num_fms, device=importance.device, dtype=importance.dtype)
        usage.index_add_(0, flat_ids, flat_importance)
        return usage / usage.sum().clamp_min(1e-8)

    def _fm_load_balance_loss(
        self,
        fm_weights_local: torch.Tensor,
        fm_ids: torch.Tensor,
        slice_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Load-balance loss over global FM IDs (not local subset positions).

        Aggregates per-FM gating mass across batch and slices, then penalizes
        deviation from the target mass each FM should receive when present.
        The target is defined per sample: for a K-FM subset, each selected FM
        should receive 1/K of the local softmax mass on that sample, not the average mass of all FMs.

        Args:
            fm_weights_local: [B, num_slices, K] FM weights per slice.
            fm_ids: [B, K] global FM IDs for the selected FMs.
            slice_mask: [B, num_slices] validity mask (True = valid slices), optional.
        """
        num_fms = self.num_fms if self.num_fms is not None else int(fm_ids.max().item()) + 1
        device = fm_weights_local.device

        if slice_mask is not None:
            slice_valid = slice_mask.to(device=device, dtype=fm_weights_local.dtype)  # [B, D]
            denom = slice_valid.sum(dim=1).clamp_min(1.0)  # [B]
            importance = (fm_weights_local * slice_valid.unsqueeze(-1)).sum(dim=1) / denom[:, None]
        else:
            importance = fm_weights_local.mean(dim=1)  # [B, K] average over slices
        valid = torch.ones_like(importance)

        per_sample_k = valid.sum(dim=1).clamp_min(1.0)  # [B]
        target_local = valid / per_sample_k[:, None]    # [B, K]

        flat_ids = fm_ids.to(device=device, dtype=torch.long).reshape(-1)  # [B*K]
        flat_importance = (importance * valid).reshape(-1)                 # [B*K]
        flat_target = target_local.reshape(-1)                              # [B*K]
        flat_valid = valid.reshape(-1)                                     # [B*K]

        usage = torch.zeros(num_fms, device=device, dtype=importance.dtype)
        target = torch.zeros(num_fms, device=device, dtype=importance.dtype)
        count = torch.zeros(num_fms, device=device, dtype=importance.dtype)
        usage.index_add_(0, flat_ids, flat_importance)
        target.index_add_(0, flat_ids, flat_target)
        count.index_add_(0, flat_ids, flat_valid)

        active = count > 0
        k_active = active.sum()
        if k_active <= 1:
            return torch.zeros((), device=device, dtype=importance.dtype)
        mean_usage = usage[active] / count[active].clamp_min(1e-8)
        mean_target = target[active] / count[active].clamp_min(1e-8)
        return k_active.to(importance.dtype) * ((mean_usage - mean_target) ** 2).sum()

    @staticmethod
    def _router_z_loss(
        router_logits: torch.Tensor,
        slice_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Router z-loss: mean(logsumexp(logits, dim=-1) ** 2). Regularizes logit magnitude."""
        per_slice = torch.logsumexp(router_logits.float(), dim=-1) ** 2  # [B, D]
        if slice_mask is None:
            return per_slice.mean()
        valid = slice_mask.to(device=router_logits.device, dtype=per_slice.dtype)
        return (per_slice * valid).sum() / valid.sum().clamp_min(1.0)

    def _collect_router_losses(self, stats_list: list) -> dict:
        """
        Aggregate router load-balance / z-loss / entropy_for_loss over the provided FM stats.

        Args:
            stats_list: list of FM stats dicts (e.g. from each contrastive view).

        Returns:
            dict with loss_router_balance, loss_router_z, loss_router_entropy (scalars),
            and fm_weight_entropy (detached mean for logging).
        """
        device = next(self.parameters()).device
        balance = torch.zeros((), device=device)
        z_loss = torch.zeros((), device=device)
        entropy_for_loss = torch.zeros((), device=device)
        fm_weight_entropy = torch.zeros((), device=device)
        fm_usage = None
        if self.num_fms is not None:
            fm_usage = torch.zeros(self.num_fms, device=device)
        n = 0
        for stats in stats_list:
            if stats is None:
                continue
            weights = stats["fm_weights_local"]
            fm_ids = stats["fm_ids"]
            slice_mask = stats.get("slice_mask")
            balance = balance + self._fm_load_balance_loss(
                weights, fm_ids, slice_mask
            )
            if stats.get("router_logits") is not None:
                z_loss = z_loss + self._router_z_loss(stats["router_logits"], slice_mask)
            entropy = stats.get("fm_weight_entropy")
            if entropy is not None:
                if slice_mask is not None:
                    valid = slice_mask.to(device=entropy.device, dtype=entropy.dtype)
                    mean_entropy = (entropy * valid).sum() / valid.sum().clamp_min(1.0)
                else:
                    mean_entropy = entropy.mean()
                entropy_for_loss = entropy_for_loss + mean_entropy
                fm_weight_entropy = fm_weight_entropy + mean_entropy.detach()
            if fm_usage is not None:
                fm_usage = fm_usage + self.aggregate_fm_usage(
                    weights, fm_ids, self.num_fms, slice_mask=slice_mask
                )
            n += 1
        denom = max(n, 1)
        out = {
            "loss_router_balance": balance / denom,
            "loss_router_z": z_loss / denom,
            "loss_router_entropy": entropy_for_loss / denom,
            "fm_weight_entropy": fm_weight_entropy / denom,
        }
        if fm_usage is not None:
            out["fm_usage"] = fm_usage / denom
        return out

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
        fm_ids_1: torch.Tensor | None = None,
        fm_ids_2: torch.Tensor | None = None,
    ) -> torch.Tensor | dict:
        """        
        Args:
            x1: First view features [B, num_slices, feature_dim] (pair) or
                [B, K, num_slices, (regions,) feature_dim] (subset).
            x2: Second view features, same layout as x1.
            input_feature_dims_1: Original feature dims for x1 ([B] pair, [B, K] subset).
            input_feature_dims_2: Original feature dims for x2 ([B] pair, [B, K] subset).
            m: Momentum parameter for momentum encoder update. Default: 0.99.
            seq_lengths: Actual sequence lengths with shape [B] for masking padded positions.
                         Shared across both views since positive pairs come from the same exam/plane.
            labels: Optional multi-label tensor [B, C] for semi-supervised contrastive learning.
            has_label: Optional boolean tensor [B] indicating which samples have classification labels.
            physical_positions: Normalized relative slice depth in [0, 1] [B, num_slices] for sinusoidal PE.
            fm_ids_1, fm_ids_2: Global FM IDs [B, K] per view (subset mode only).
            
        Returns:
            Scalar contrastive loss, or a dict with auxiliary losses/metrics when
            router losses are active.
        """
        subset_mode = fm_ids_1 is not None
        pe_kwargs = dict(physical_positions=physical_positions)

        q1 = self.predictor(self.base_encoder(
            x1, input_feature_dims=input_feature_dims_1, seq_lengths=seq_lengths,
            fm_ids=fm_ids_1, **pe_kwargs,
        ))
        stats1 = self.base_encoder._last_fm_stats if subset_mode else None
        q2 = self.predictor(self.base_encoder(
            x2, input_feature_dims=input_feature_dims_2, seq_lengths=seq_lengths,
            fm_ids=fm_ids_2, **pe_kwargs,
        ))
        stats2 = self.base_encoder._last_fm_stats if subset_mode else None
       
        with torch.no_grad():
            self._update_momentum_encoder(m=m)
            k1 = self.momentum_encoder(
                x1, input_feature_dims=input_feature_dims_1, seq_lengths=seq_lengths,
                fm_ids=fm_ids_1, **pe_kwargs,
            )
            k2 = self.momentum_encoder(
                x2, input_feature_dims=input_feature_dims_2, seq_lengths=seq_lengths,
                fm_ids=fm_ids_2, **pe_kwargs,
            )

        loss_infonce = (
            self.contrastive_loss(q1, k2, labels=labels, has_label=has_label) + 
            self.contrastive_loss(q2, k1, labels=labels, has_label=has_label)
        )

        if subset_mode:
            return self._router_aux_loss(loss_infonce, [stats1, stats2])

        return loss_infonce

    def _router_aux_loss(
        self,
        loss_infonce: torch.Tensor,
        stats_list: list,
    ) -> torch.Tensor | dict:
        """
        Combine InfoNCE with optional router auxiliary losses for subset FM mode.

        - Router: total = InfoNCE + w_balance * balance + w_z * z + w_entropy * entropy.
          The entropy loss term penalizes high entropy, encouraging sharper routing
          when enabled (default off).
        - Attention / avg_pool: InfoNCE only; log fm_weight_entropy for monitoring. 
          Router aux weights (router_load_balance_weight, etc.) apply only when fm_pooling='router'.
        """
        router_losses = self._collect_router_losses(stats_list)

        if self.fm_pooling != "router":
            out = {
                "loss": loss_infonce,
                "loss_infonce": loss_infonce.detach(),
                "fm_weight_entropy": router_losses["fm_weight_entropy"],
            }
            if "fm_usage" in router_losses:
                out["fm_usage"] = router_losses["fm_usage"].detach()
            return out

        total_loss = (
            loss_infonce
            + self.router_load_balance_weight * router_losses["loss_router_balance"]
            + self.router_z_loss_weight * router_losses["loss_router_z"]
            + self.router_entropy_weight * router_losses["loss_router_entropy"]
        )
        out = {
            "loss": total_loss,
            "loss_infonce": loss_infonce.detach(),
            "loss_router_balance": router_losses["loss_router_balance"].detach(),
            "loss_router_z": router_losses["loss_router_z"].detach(),
            "loss_router_entropy": router_losses["loss_router_entropy"].detach(),
            "fm_weight_entropy": router_losses["fm_weight_entropy"],
        }
        if "fm_usage" in router_losses:
            out["fm_usage"] = router_losses["fm_usage"].detach()
        return out
    
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

        return loss_infonce

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
        fm_ids_1: torch.Tensor | None = None,
        fm_ids_2: torch.Tensor | None = None,
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
            physical_positions: Normalized relative slice depth in [0, 1] for sinusoidal PE.
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
            Scalar contrastive loss, or a dict with router auxiliary losses in subset mode.
        """
        subset_mode = fm_ids_1 is not None or fm_ids_2 is not None
        if (fm_ids_1 is None) != (fm_ids_2 is None):
            raise ValueError("Subset FM mode requires both fm_ids_1 and fm_ids_2.")
        if self.fm_pooling == "router" and not subset_mode:
            raise ValueError(
                f"fm_pooling='{self.fm_pooling}' requires subset FM mode with fm_ids. "
                "Use ssl_fm_mode='subset' or switch fm_pooling='avg_pool'."
            )
        if use_packed:
            if fm_ids_1 is not None:
                raise NotImplementedError(
                    "Packed subset FM mode is not supported yet. Use padded subset batches "
                    "(use_packed=False) for fm_pooling router SSL."
                )
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
                fm_ids_1=fm_ids_1,
                fm_ids_2=fm_ids_2,
                **kwargs,
            )
    
