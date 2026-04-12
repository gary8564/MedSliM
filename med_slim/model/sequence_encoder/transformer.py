"""
Adapted from https://github.com/mueller-franzes/MST/blob/main/mst/data/datasets/augmentations/augmentations_3d.py
Müller-Franzes, Gustav, Firas Khader, Robert Siepmann, Tianyu Han, Jakob Nikolas Kather, Sven Nebelung and Daniel Truhn. 
"Medical Slice Transformer: Improved Diagnosis and Explainability on 3D Medical Images with DINOv2."
ArXiv abs/2411.15802 (2024).
"""
from typing import Optional, Union, Callable, Tuple
import math
import torch
from torch import Tensor
import torch.nn.functional as F 
import torch.nn as nn
from torch.nn import Module, Dropout, Linear, LayerNorm
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
from torch.nn.parameter import Parameter
from torch.overrides import has_torch_function, handle_torch_function

from torch.nn.functional import _in_projection, _mha_shape_check, _in_projection_packed, _canonical_mask, _none_or_dtype
from torch.nn.functional import pad, softmax, linear, scaled_dot_product_attention, dropout

from .rotary_embedding_torch import RotaryEmbedding, AttentionLiereRotator

from flash_attn import flash_attn_varlen_func

from .rotary_embedding_torch import apply_rotary_emb

def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")



def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Optional[Tensor],
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    average_attn_weights: bool = True,
    is_causal: bool = False,
    rotary_positional_encoding: Optional[str] = None
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""See PyTorch implementation 
     Adds:  rotary_positional_encoding
  
    """
    tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v, out_proj_weight, out_proj_bias)
    if has_torch_function(tens_ops):
        return handle_torch_function(
            multi_head_attention_forward,
            tens_ops,
            query,
            key,
            value,
            embed_dim_to_check,
            num_heads,
            in_proj_weight,
            in_proj_bias,
            bias_k,
            bias_v,
            add_zero_attn,
            dropout_p,
            out_proj_weight,
            out_proj_bias,
            training=training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
            is_causal=is_causal,
            use_separate_proj_weight=use_separate_proj_weight,
            q_proj_weight=q_proj_weight,
            k_proj_weight=k_proj_weight,
            v_proj_weight=v_proj_weight,
            static_k=static_k,
            static_v=static_v,
            average_attn_weights=average_attn_weights,
        )

    is_batched = _mha_shape_check(query, key, value, key_padding_mask, attn_mask, num_heads)

    # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
    # is batched, run the computation and before returning squeeze the
    # batch dimension so that the output doesn't carry this temporary batch dimension.
    if not is_batched:
        # unsqueeze if the input is unbatched
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape

    key_padding_mask = _canonical_mask(
        mask=key_padding_mask,
        mask_name="key_padding_mask",
        other_type=_none_or_dtype(attn_mask),
        other_name="attn_mask",
        target_type=query.dtype
    )

    if is_causal and attn_mask is None:
        raise RuntimeError(
            "Need attn_mask if specifying the is_causal hint. "
            "You may use the Transformer module method "
            "`generate_square_subsequent_mask` to create this mask."
        )

    if is_causal and key_padding_mask is None and not need_weights:
        # when we have a kpm or need weights, we need attn_mask
        # Otherwise, we use the is_causal hint go as is_causal
        # indicator to SDPA.
        attn_mask = None
    else:
        attn_mask = _canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        if key_padding_mask is not None:
            # We have the attn_mask, and use that to merge kpm into it.
            # Turn off use of is_causal hint, as the merged mask is no
            # longer causal.
            is_causal = False

    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(embed_dim, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = embed_dim.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert key.shape[:2] == value.shape[:2], \
            f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    # Compute in-projection
    if not use_separate_proj_weight:
        assert in_proj_weight is not None, "use_separate_proj_weight is False but in_proj_weight is None"
        q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
        assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
        assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

    # prep attention mask

    if attn_mask is not None:
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    # Reshape q, k, v for multi-head attention (batch first)
    q = q.view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is None:
        k = k.view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # Static k/v bypass the in-projection; validate shapes match expected head layout
        assert static_k.size(0) == bsz * num_heads, \
            f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
        assert static_k.size(2) == head_dim, \
            f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        k = static_k
    if static_v is None:
        v = v.view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # Static k/v bypass the in-projection; validate shapes match expected head layout
        assert static_v.size(0) == bsz * num_heads, \
            f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
        assert static_v.size(2) == head_dim, \
            f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        v = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len), \
            f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
            expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        else:
            attn_mask = attn_mask + key_padding_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    # Calculate attention weights and output projection

    if rotary_positional_encoding is not None:
        q = rotary_positional_encoding(q.view(bsz, num_heads, tgt_len, head_dim)).view(bsz*num_heads, tgt_len, head_dim)
        k = rotary_positional_encoding(k.view(bsz, num_heads, src_len, head_dim)).view(bsz*num_heads, src_len, head_dim)

    if need_weights:
        B, Nt, E = q.shape
        q_scaled = q * math.sqrt(1.0 / float(E))

        assert not (is_causal and attn_mask is None), "is_causal without an explicit attn_mask is not supported when need_weights=True"

        if attn_mask is not None:
            attn_output_weights = torch.baddbmm(attn_mask, q_scaled, k.transpose(-2, -1))
        else:
            attn_output_weights = torch.bmm(q_scaled, k.transpose(-2, -1))
        attn_output_weights = softmax(attn_output_weights, dim=-1)
        if dropout_p > 0.0:
            attn_output_weights = dropout(attn_output_weights, p=dropout_p)

        attn_output = torch.bmm(attn_output_weights, v)

        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len * bsz, embed_dim)
        attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))

        # optionally average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        if average_attn_weights:
            attn_output_weights = attn_output_weights.mean(dim=1)

        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
            attn_output_weights = attn_output_weights.squeeze(0)
        return attn_output, attn_output_weights
    else:
        # attn_mask can be either (L,S) or (N*num_heads, L, S)
        # if attn_mask's shape is (1, L, S) we need to unsqueeze to (1, 1, L, S)
        # in order to match the input for SDPA of (N, num_heads, L, S)
        if attn_mask is not None:
            if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(0)
            else:
                attn_mask = attn_mask.view(bsz, num_heads, -1, src_len)

        q = q.view(bsz, num_heads, tgt_len, head_dim)
        k = k.view(bsz, num_heads, src_len, head_dim)
        v = v.view(bsz, num_heads, src_len, head_dim)

        attn_output = scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)

        attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
        return attn_output, None





class MultiheadAttention(nn.MultiheadAttention):
    r"""Modified PyTorch MultiheadAttention

    # Changes:
    * allows to use Rotary Position Encoding
    """

    def __init__(self, embed_dim, num_heads, dropout=0, bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None, rotary_positional_encoding=None) -> None:
        super().__init__(embed_dim, num_heads, dropout, bias, add_bias_kv, add_zero_attn, kdim, vdim, batch_first, device, dtype)
        if rotary_positional_encoding == 'RoPE':
            self.rotary_positional_encoding = RotaryEmbedding(        
                dim=self.head_dim,
                custom_freqs = None,
                freqs_for = 'lang', # 'pixel', 'const'
                theta = 256, # only relevant for lang 
                max_freq = 32, # only relevant for pixel
                num_freqs = 1, # only relevant for const
                learned_freq = False,
                use_xpos = False,
                xpos_scale_base = 512,
                interpolate_factor = 1.,
                theta_rescale_factor = 1.,
                seq_before_head_dim = False,
                cache_if_possible = True 
            )
            self.rotary_positional_encoding_func = self.rotary_positional_encoding.rotate_queries_or_keys
        elif rotary_positional_encoding == 'LiRE':
            self.rotary_positional_encoding = AttentionLiereRotator(
                head_dim=self.head_dim,
                liere_block_size=self.head_dim//2, # Note: head_dim = (N * liere_block_size)
                spacial_dims=1,
                axes_length=33, # Sequence length
                num_heads=num_heads
            )
            self.rotary_positional_encoding_func = self.rotary_positional_encoding.rotate_queries_or_keys
        elif rotary_positional_encoding is None:
            self.rotary_positional_encoding_func = None
        else:
            raise ValueError(f"Unkown parameter {rotary_positional_encoding} for rotary_positional_encoding")



    def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[Tensor] = None,
            average_attn_weights: bool = True,
            is_causal : bool = False) -> Tuple[Tensor, Optional[Tensor]]:
       

        is_batched = query.dim() == 3

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype
        )

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )


        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))



        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
                rotary_positional_encoding=self.rotary_positional_encoding_func)
        else:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
                rotary_positional_encoding=self.rotary_positional_encoding_func)
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

    
    

class TransformerEncoderLayer(Module):
    r"""Custom PyTorch TransformerEncoderLayer 

    Changes:
    - Rotary embeddings are applied to the key/query matrices.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectively. Otherwise it's done after. Default: ``False`` (after).
        bias: If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
            bias. Default: ``True``.

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)


    """

    __constants__ = ['norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None, rotary_positional_encoding=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout,
                                            bias=bias, batch_first=batch_first,
                                            rotary_positional_encoding=rotary_positional_encoding,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu


    def forward(
            self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: bool = False) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: If specified, applies a causal mask as ``src mask``.
                Default: ``False``.
                Warning:
                ``is_causal`` provides a hint that ``src_mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Shape:
            see the docs in :class:`~torch.nn.Transformer`.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )


        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False, is_causal=is_causal)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class VarlenMultiheadAttention(nn.Module):
    """
    Multi-head attention for packed/variable-length sequences using FlashAttention.
    
    Input shape: [total_seq_len, embed_dim] (packed sequences)
    
    References:
        - https://github.com/Dao-AILab/flash-attention
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        dropout: float = 0.0,
        bias: bool = True,
        rotary_positional_encoding: Optional[str] = None,
        device=None, 
        dtype=None
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # QKV projection
        self.qkv_proj = Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        
        # Rotary positional encoding
        if rotary_positional_encoding == 'RoPE':
            self.rotary_emb = RotaryEmbedding(
                dim=self.head_dim,
                freqs_for='lang',
                theta=256,
                cache_if_possible=True
            )
        else:
            self.rotary_emb = None
    
    def forward(
        self, 
        x: Tensor, 
        cu_seqlens: Tensor, 
        max_seqlen: int,
        is_causal: bool = False
    ) -> Tensor:
        """
        Forward pass for variable-length attention using FlashAttention.
        
        Args:
            x: Packed input tensor [total_seq_len, embed_dim]
            cu_seqlens: Cumulative sequence lengths [batch_size + 1], int32
            max_seqlen: Maximum sequence length in the batch
            is_causal: Whether to apply causal masking
            
        Returns:
            Output tensor [total_seq_len, embed_dim]
        """
        total_seq_len = x.shape[0]
        batch_size = cu_seqlens.shape[0] - 1
        
        # Project to Q, K, V: [total_seq_len, 3 * embed_dim]
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention: [total_seq_len, num_heads, head_dim]
        q = q.view(total_seq_len, self.num_heads, self.head_dim)
        k = k.view(total_seq_len, self.num_heads, self.head_dim)
        v = v.view(total_seq_len, self.num_heads, self.head_dim)
        
        # Apply RoPE with per-slice positions
        # Each slice's positions start from 0, not global positions
        if self.rotary_emb is not None:
            # Compute per-slice positions for packed sequences
            # For each token: position = global_index - cu_seqlens[batch_idx]
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]  # [batch_size]
            positions = (
                torch.arange(total_seq_len, device=x.device, dtype=torch.float32) 
                - cu_seqlens[:-1].to(x.device, dtype=torch.float32).repeat_interleave(lengths)
            )
            
            # Generate frequencies for these positions: [total_seq_len, head_dim]
            freqs = self.rotary_emb.forward(positions, seq_len=total_seq_len)
            
            # Reshape freqs for broadcasting: [total_seq_len, 1, head_dim]
            # This broadcasts across all heads
            freqs = freqs.unsqueeze(1)
            
            # Apply RoPE to Q and K: [total_seq_len, num_heads, head_dim]
            q = apply_rotary_emb(freqs, q, seq_dim=0)
            k = apply_rotary_emb(freqs, k, seq_dim=0)
        
        # FlashAttention for variable-length sequences
        out = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=self.dropout if self.training else 0.0,
            causal=is_causal,
        )
        
        # Reshape back: [total_seq_len, embed_dim]
        out = out.view(total_seq_len, self.embed_dim)
        out = self.out_proj(out)
        
        return out


class VarlenTransformerEncoderLayer(Module):
    """
    Transformer encoder layer for packed/variable-length sequences using FlashAttention.
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.gelu,
        layer_norm_eps: float = 1e-5,
        norm_first: bool = True,
        bias: bool = True,
        rotary_positional_encoding: Optional[str] = None,
        device=None,
        dtype=None
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        self.self_attn = VarlenMultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            bias=bias,
            rotary_positional_encoding=rotary_positional_encoding,
            **factory_kwargs
        )
        
        # Feedforward network
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)
        
        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        
        # Activation
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)
        self.activation = activation
    
    def forward(
        self,
        src: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: int,
        is_causal: bool = False
    ) -> Tensor:
        """
        Forward pass for packed sequences.
        
        Args:
            src: Packed input tensor [total_seq_len, d_model]
            cu_seqlens: Cumulative sequence lengths [batch_size + 1], int32
            max_seqlen: Maximum sequence length in the batch
            is_causal: Whether to apply causal masking
            
        Returns:
            Output tensor [total_seq_len, d_model]
        """
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), cu_seqlens, max_seqlen, is_causal)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, cu_seqlens, max_seqlen, is_causal))
            x = self.norm2(x + self._ff_block(x))
        return x
    
    def _sa_block(
        self, 
        x: Tensor, 
        cu_seqlens: Tensor, 
        max_seqlen: int,
        is_causal: bool
    ) -> Tensor:
        x = self.self_attn(x, cu_seqlens, max_seqlen, is_causal)
        return self.dropout1(x)
    
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class VarlenTransformerEncoder(Module):
    """
    Transformer encoder stack for packed/variable-length sequences using FlashAttention.
    
    Usage:
        >>> encoder = VarlenTransformerEncoder(d_model=768, nhead=8, num_layers=6)
        >>> # Packed input: [total_seq_len, d_model]
        >>> # cu_seqlens: [batch_size + 1] marking sequence boundaries
        >>> out = encoder(x_packed, cu_seqlens, max_seqlen)
    """
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = True,
        rotary_positional_encoding: Optional[str] = None,
        device=None,
        dtype=None
    ):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        self.layers = nn.ModuleList([
            VarlenTransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                norm_first=norm_first,
                rotary_positional_encoding=rotary_positional_encoding,
                **factory_kwargs
            )
            for _ in range(num_layers)
        ])
        self.norm = LayerNorm(d_model, **factory_kwargs)
    
    def forward(
        self,
        src: Tensor,
        cu_seqlens: Tensor,
        max_seqlen: int,
        is_causal: bool = False
    ) -> Tensor:
        """
        Forward pass through all encoder layers.
        
        Args:
            src: Packed input [total_seq_len, d_model]
            cu_seqlens: Cumulative sequence lengths [batch_size + 1]
            max_seqlen: Maximum sequence length
            is_causal: Whether to use causal attention
            
        Returns:
            Encoded output [total_seq_len, d_model]
        """
        output = src
        for layer in self.layers:
            output = layer(output, cu_seqlens, max_seqlen, is_causal)
        return self.norm(output)