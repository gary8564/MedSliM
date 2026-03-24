"""
Lightweight predictor for Masked Slice Prediction (MSP).

Takes the student encoder's hidden states at masked positions, adds learnable
positional embeddings, and maps them through a narrow residual MLP to predict
the teacher encoder's latent representations.

The predictor is intentionally kept small (~1-2 M parameters) so that the
representation quality is driven by the encoder rather than the predictor
capacity.

References:
    [1] Assran et al., "Self-Supervised Learning from Images with a Joint-Embedding
        Predictive Architecture", CVPR 2023. https://arxiv.org/abs/2301.08243
    [2] Mur-Labadia et al., "V-JEPA 2.1: Unlocking Dense Features in Video
        Self-Supervised Learning", 2026. https://arxiv.org/abs/2603.14482
"""

import torch
import torch.nn as nn


class MSPPredictor(nn.Module):
    """
    Residual MLP predictor with learnable positional embeddings.

    Architecture (per block):
        LayerNorm → Linear(embed_dim, hidden_dim) → GELU → Dropout → Linear(hidden_dim, embed_dim)
    with a residual connection around each block.

    Args:
        embed_dim: Encoder hidden dimension (input and output size).
        hidden_dim: Bottleneck width inside each residual block.
                    Defaults to embed_dim // 4.
        num_layers: Number of residual MLP blocks.
        max_seq_len: Maximum slice-sequence length (for positional embedding table).
        dropout: Dropout rate inside residual blocks.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int | None = None,
        num_layers: int = 2,
        max_seq_len: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // 4

        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    nn.Linear(embed_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                    nn.Linear(hidden_dim, embed_dim),
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, h: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Hidden states at positions to predict, [N, embed_dim].
            positions: Slice-position indices [N] (int, used for positional
                embedding look-up).

        Returns:
            Predicted representations [N, embed_dim].
        """
        h = h + self.pos_embed(positions)
        for block in self.blocks:
            h = h + block(h)
        return self.norm(h)
