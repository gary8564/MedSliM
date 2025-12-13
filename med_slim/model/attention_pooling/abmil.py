"""
Adapted from: 
[1] https://github.com/mahmoodlab/MADELEINE/blob/main/core/models/abmil.py
Guillaume Jaume, Anurag Jayant Vaidya, Andrew Zhang,
Andrew H Song, Richard J. Chen, Sharifa Sahai, Dandan
Mo, Emilio Madrigal, Long Phi Le, and Mahmood Faisal.
Multistain pretraining for slide representation learning in
pathology. In European Conference on Computer Vision.
Springer, 2024.
[2] https://github.com/AMLab-Amsterdam/AttentionDeepMIL/blob/master/model.py
Ilse, Maximilian, Jakub M. Tomczak, and Max Welling 
Attention-based deep multiple instance learning. 
arXiv.Org. February 13, 2018. https://arxiv.org/abs/1802.04712.
"""
import torch 
from torch import nn
import torch.nn.functional as F

class BatchedABMIL(nn.Module):

    def __init__(self, input_dim=1024, hidden_dim=256, dropout=False, n_heads=1, activation='softmax'):
        """
        Attention Network with Sigmoid Gating (3 fc layers). Supports batching.
        
        args:
            input_dim (int): input feature dimension
            hidden_dim (int): hidden layer dimension
            dropout (bool): whether to use dropout (p = 0.25)
            n_heads (int): number of attention heads
            activation (str): activation function
        """
        super(BatchedABMIL, self).__init__()

        self.activation = activation
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # attention_V = tanh(V h_k)
        self.attention_V = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        ])

        # attention_U = sigmoid(U h_k)
        self.attention_U = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        ])
        
        if dropout:
            self.attention_V.append(nn.Dropout(0.25))
            self.attention_U.append(nn.Dropout(0.25))

        self.attention_V = nn.Sequential(*self.attention_V)
        self.attention_U = nn.Sequential(*self.attention_U)
        # attention_W = w^T (attention_V ⊙ attention_U)
        self.attention_W = nn.Linear(hidden_dim, n_heads)

    def forward(self, x, mask=None, return_raw_attention=False):
        """
        Forward pass 
        Args:
            x (torch.Tensor): shape (bs, num_tokens, embed_dim)
            mask (torch.Tensor, optional): Boolean mask of shape (bs, num_tokens) where True indicates 
                                           valid (real) positions and False indicates padded positions.
                                           If provided, padded positions will be masked out before softmax.
            return_raw_attention (bool): whether to return the raw attention weights
        Returns:
            activated_A (torch.Tensor): Activated attention weights
            A (torch.Tensor): Raw attention weights only if return_raw_attention=True
        """
        assert len(x.shape)==3, x.shape
        a = self.attention_V(x)  # [batch_size, num_tokens, hidden_dim]
        b = self.attention_U(x)  # [batch_size, num_tokens, hidden_dim]
        A = a.mul(b)  # element-wise gated attention [batch_size, num_tokens, hidden_dim]
        A = self.attention_W(A)  # [batch_size, num_tokens, n_heads]

        # Apply mask to attention scores before activation 
        if mask is not None:
            # mask shape: [batch_size, num_tokens] with boolean values (True = valid, False = padded)
            # A shape: [batch_size, num_tokens, n_heads]
            # Expand mask to match A's shape: [batch_size, num_tokens, 1]
            mask = mask.unsqueeze(-1)  # [batch_size, num_tokens, 1]
            A = A.masked_fill(~mask, float('-inf'))

        # Based on the task, we can choose different activation functions
        # Softmax: classic MIL assumption (one/few key instances drive the decision; point to the single worst slice).
        # Others: allows multiple instances to contribute independently, not normalized (accumulate how many slices look abnormal and how strongly).
        # This is useful in 3D PET/CT scan or MRI images where imagine conditions that are diffuse, volumetric, or severity-related
        if self.activation == 'softmax':
            activated_A = F.softmax(A, dim=1) # normalize attention weights
        elif self.activation == 'leaky_relu':
            activated_A = F.leaky_relu(A) # "counting/aggregating" attention weights
        elif self.activation == 'relu':
            activated_A = F.relu(A) # "counting/aggregating" attention weights
        elif self.activation == 'sigmoid': 
            activated_A = torch.sigmoid(A) # "counting/aggregating" attention weights
        else:
            raise NotImplementedError('Activation not implemented.')

        if return_raw_attention:
            return activated_A, A

        return activated_A