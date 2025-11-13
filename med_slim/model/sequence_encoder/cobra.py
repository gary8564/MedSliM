"""
Adapted from: https://github.com/KatherLab/COBRA/blob/main/cobra/model/model.py
Lenz, Tim, Peter Neidlinger, Marta Ligero, Georg Wölflein, Marko van Treeck and Jakob Nikolas Kather. 
Unsupervised Foundation Model-Agnostic Slide-Level Representation Learning.
2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR): 30807-30817, 2024.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .mamba2 import Mamba2Enc
from med_slim.model.attention_pooling import BatchedABMIL
from einops import rearrange 


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
    normalization layer and a mamba-based encoder (Mamba2Enc). It then applies multi-head
    attention using BatchedABMIL modules to compute attention maps and aggregate the input features.
    
    Example:
        >>> model = Cobra()
        >>> # Processing random input
        >>> x = torch.randn(1, 100, 768)  # batch size: 1, sequence length: 100, feature dimension: 768
        >>> features = model(x)
    """

    def __init__(self,
                 embed_dim=768, 
                 contrast_dim=256, 
                 input_dims=[512, 768, 1024, 1152, 1376], 
                 num_heads=8,
                 layer=2,
                 dropout=0.25,
                 att_dim=256,
                 d_state=128,
                 mode="train"):
        """
        Parameters:
        embed_dim (int, optional):
            Dimensionality of the embedding vectors.
        contrast_dim (int, optional):
            Dimensionality of the contrastive features.
        input_dims (list of int, optional):
            A list of input feature dimensions. Each feature dimension corresponds to a key in the
            embedding module dictionary. Default is [384, 512, 768, 1024, 1152, 1376].
        num_heads (int, optional):
            Number of attention heads. Each head processes a slice of the embedded features.
            Default is 8.
        layers (int, optional):
            Number of layers in the Mamba2Enc encoder. Default is 2.
        dropout (float, optional):
            Dropout rate used throughout the model to prevent overfitting. Default is 0.25.
        att_dim (int, optional):
            The hidden dimensionality for the attention branch (BatchedABMIL) per attention head.
            Default is 256.
        d_state (int, optional):
            Dimensionality of the internal state in the Mamba2Enc encoder. Default is 128.
        mode (str, optional):
            'train' or 'inference'. If 'inference', the model will not use the projection layer and the original patch embeddings are used as feature representation instead of Mamba-encoded embeddings. 
            Default is 'train'.
        """
        super().__init__()
        
        assert mode in ["train", "inference"], "mode must be either 'train' or 'inference', got {mode}."
        self.mode = mode
        self.embed_dim = embed_dim

        self.embed = nn.ModuleDict({str(d): Embed(d, embed_dim) for d in input_dims})
        
        self.norm = nn.LayerNorm(embed_dim)
        
        self.mamba_enc = Mamba2Enc(embed_dim, embed_dim, n_classes=embed_dim, layer=layer, dropout=dropout, d_state=d_state)

        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout else nn.Identity(),
            nn.Linear(4 * embed_dim, contrast_dim),
            nn.BatchNorm1d(contrast_dim),
        )
    
        self.num_heads = num_heads
        self.attn = BatchedABMIL(input_dim=embed_dim, hidden_dim=att_dim, dropout=dropout, n_heads=num_heads, activation='softmax') 

    def forward(self, x, input_feature_dims=None, get_attention=False):
        """
        Forward pass through the Cobra network.
        Args:
            x (Tensor or list of Tensors):
                Input tensor with shape [batch_size, num_slices, feature_dim]. 
                Each tensor should have a feature_dim corresponding to the respective key in the embedding module.
            input_feature_dims (Tensor, optional):
                Tensor of shape [batch_size] containing the feature dimensions of the input.
                Default is None.
            get_attention (bool, optional):
                If True, the method returns the computed attention matrix rather than the aggregated features.
                Default is False.
        Returns:
            Tensor:
                If get_attention is True, returns the attention matrix computed from the input.
                Otherwise, returns the aggregated feature representation obtained after applying the attention mechanism.
        Raises:
            AssertionError:
                If the dimensions of the embedded features do not match the expected sizes during
                concatenation or aggregation, assertions will be raised to signal the discrepancy.
        """
        # Foundation model feature embedding stage
        if input_feature_dims is not None:
            assert len(x)==len(input_feature_dims), f"Batch size mismatch between input x and input_feature_dims"
            logits = torch.concat([self.embed[str(input_feature_dims[i].item())](x[i,:,:input_feature_dims[i].item()]).unsqueeze(0) for i in range(len(x))], dim=0) # [B, num_slices, embed_dim]
        else:
            logits = self.embed[str(x.shape[-1])](x) # [B, num_slices, embed_dim]
        # Mamba encoder + LayerNorm
        h = self.norm(self.mamba_enc(logits)) # [B, num_slices, embed_dim]
        
        # Multi-head attention mechanism stage
        activated_A, A_raw = self.attn(h, return_raw_attention=True) # [B, num_slices, num_heads]
        A = activated_A.mean(dim=-1, keepdim=True) # Average over heads -> [B, num_slices, 1]
        A = A.transpose(1, 2) # [B, 1, num_slices]
        
        if get_attention:
            # return the bag-level attention map
            return A
        
        # Training phase: MIL pooling over feature embedding after Mamba-encoder
        # Inference phase: MIL pooling over original input features
        if self.mode == "train":
            h = torch.bmm(A, h).squeeze() # [B, embed_dim]
            feats = self.proj(h)
            
        else:
            feats = torch.bmm(A, logits).squeeze()    # [B, embed_dim]
        
        assert len(feats.shape)==2, feats.shape
        return feats