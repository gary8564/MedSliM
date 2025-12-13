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
from med_slim.model.sequence_encoder.cobra import Cobra

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    # If distributed is not initialized, just return the input tensor
    if (not torch.distributed.is_available()) or (not torch.distributed.is_initialized()):
        return tensor
    world_size = torch.distributed.get_world_size()
    tensors_gather = [torch.ones_like(tensor) for _ in range(world_size)] # world_size = number of GPUs
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

class MoCo(nn.Module): 
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self,
                 embed_dim, 
                 contrast_dim, 
                 input_dims=[384, 512, 768, 1024, 1152, 1376],
                 num_heads=8, 
                 num_mamba_layers=2, 
                 T=0.2,
                 dropout=0.25,
                 att_dim=256,
                 d_state=64):
        """
        Parameters:
        embed_dim (int):
            Dimensionality of the embedding vectors.
        contrast_dim (int):
            Dimensionality of the contrastive features.
        input_dims (list of int, optional):
            A list of input feature dimensions. Each feature dimension corresponds to a key in the
            embedding module dictionary. Default is [384, 512, 768, 1024, 1152, 1376].
        num_heads (int, optional):
            Number of attention heads. Each head processes a slice of the embedded features.
            Default is 8.
        num_mamba_layers (int, optional):
            Number of layers in the Mamba2Enc encoder. Default is 2.
        gpu_id (int, optional):
            GPU ID to use for the model. Default is 0.
        T (float, optional):
            Softmax temperature parameter for the contrastive loss. Default is 0.2.
        dropout (float, optional):
            Dropout rate used throughout the model to prevent overfitting. Default is 0.25.
        att_dim (int, optional):
            The hidden dimensionality for the attention mechanism (BatchedABMIL) per attention head.
            Default is 256.
        d_state (int, optional):
            Dimensionality of the internal state in the Mamba2Enc encoder. Default is 64.
        """
        super().__init__()

        self.T = T
        self.base_encoder = Cobra(embed_dim, contrast_dim, input_dims, num_heads, layer=num_mamba_layers, dropout=dropout,
                              att_dim=att_dim, d_state=d_state)
        self.momentum_encoder = Cobra(embed_dim, contrast_dim, input_dims, num_heads, layer=num_mamba_layers, dropout=None,
                                  att_dim=att_dim,d_state=d_state)
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


    def forward(self, 
                x1, 
                x2, 
                *, 
                input_feature_dims_1: torch.Tensor | None = None, 
                input_feature_dims_2: torch.Tensor | None = None, 
                seq_lengths_1: torch.Tensor | None = None,
                seq_lengths_2: torch.Tensor | None = None,
                m: float = 0.99):
        """
        Args:
            x1 (Tensor): First view of the input images.
            x2 (Tensor): Second view of the input images.
            input_feature_dims_1 (Tensor, optional): Original feature dims per-sample for x1 (before padding).
            input_feature_dims_2 (Tensor, optional): Original feature dims per-sample for x2 (before padding).
            seq_lengths_1 (Tensor, optional): Actual sequence lengths per-sample for x1 (before subsampling/zero-padding).
            seq_lengths_2 (Tensor, optional): Actual sequence lengths per-sample for x2 (before subsampling/zero-padding).
            m (float, optional): Momentum parameter. Default is 0.99.
        Returns:
            Tensor: Contrastive loss.
        """
        # Compute the contrastive features 
        q1 = self.predictor(self.base_encoder(x1, input_feature_dims=input_feature_dims_1, seq_lengths=seq_lengths_1))
        q2 = self.predictor(self.base_encoder(x2, input_feature_dims=input_feature_dims_2, seq_lengths=seq_lengths_2))
       
        with torch.no_grad():  # no gradient
            self._update_momentum_encoder(m=m) # update the momentum encoder

            # Compute the contrastive features for the momentum encoder as targets
            k1 = self.momentum_encoder(x1, input_feature_dims=input_feature_dims_1, seq_lengths=seq_lengths_1)
            k2 = self.momentum_encoder(x2, input_feature_dims=input_feature_dims_2, seq_lengths=seq_lengths_2)

        return self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1) # return the contrastive loss
    
    def contrastive_loss(self, q, k):
        # normalize
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        # gather all targets
        k = concat_all_gather(k) # shape [world_size * batch_size, contrast_dim]
        # Einstein sum is more intuitive
        logits = torch.einsum('nc, mc -> nm', [q, k]) / self.T # shape [batch_size, world_size * batch_size]
        N = logits.shape[0]  # batch size per GPU = world_size * batch_size
        rank = torch.distributed.get_rank() if (torch.distributed.is_available() and torch.distributed.is_initialized()) else 0
        labels = (torch.arange(N, dtype=torch.long, device=logits.device) + N * rank) # for query i, the correct class index is i (or i + offset) (N * rank is for multi-GPU setting)
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)