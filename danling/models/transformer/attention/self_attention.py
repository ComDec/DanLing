import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SelfAttention(nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_
    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
    where :math:`head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)`.
    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        k_dim: total number of features in key. Default: None.
        v_dim: total number of features in value. Default: None.
        batch_first: If ``False``, then the input and output tensors are provided
            as (seq, batch, feature). Default: ``True`` (batch, seq, feature).
    Note that if :attr:`k_dim` and :attr:`v_dim` are None, they will be set
    to :attr:`embed_dim` such that query, key, and value have the same
    number of features.
    Examples::
        >>> multihead_attn = dl.models.SelfAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """
    __constants__ = ["batch_first"]

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout: Optional[float] = 0.0,
        scale_factor: Optional[float] = 1.0,
        bias: Optional[bool] = True,
        batch_first: Optional[bool] = True,
        **kwargs: Optional[Dict[str, Any]],
    ) -> None:
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.scale_factor = scale_factor
        self.batch_first = batch_first
        self.head_dim = self.embed_dim // self.num_heads
        self.scaling = float(self.head_dim * self.scale_factor) ** -0.5
        if not self.head_dim * self.num_heads == self.embed_dim:
            raise ValueError(
                f"embed_dim {self.embed_dim} not divisible by num_heads {self.num_heads}"
            )

        self.in_proj = nn.Linear(
            self.embed_dim, self.embed_dim + self.k_dim + self.v_dim, bias=bias
        )
        self.dropout = nn.Dropout(attn_dropout)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        if self.in_proj.bias is not None:
            nn.init.constant_(self.in_proj.bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_bias: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: Optional[bool] = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""
        Args:
            query, key, value: map a query and a set of key-value pairs to an output.
                See "Attention Is All You Need" for more details.
            attn_bias: 2D or 3D mask that add bias to attention output weights. Used for relative positional embedding.
                A 2D bias will be broadcasted for all the batches while a 3D mask allows to specify a different mask for
                the entries of each batch.
            attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
                the batches while a 3D mask allows to specify a different mask for the entries of each batch.
            key_padding_mask: if provided, specified padding elements in the key will
                be ignored by the attention. When given a binary mask and a value is True,
                the corresponding value on the attention layer will be ignored. When given
                a byte mask and a value is non-zero, the corresponding value on the attention
                layer will be ignored
            need_weights: output attn_output_weights.
        Shapes for inputs:
            - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
                the embedding dimension. :math:`(N, L, E)` if ``batch_first`` is ``True``.
            - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
                the embedding dimension. :math:`(N, S, E)` if ``batch_first`` is ``True``.
            - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
                the embedding dimension. :math:`(N, S, E)` if ``batch_first`` is ``True``.
            - attn_bias: if a 2D mask: :math:`(L, S)` where L is the target sequence length, S is the
                source sequence length.
                If a 3D mask: :math:`(N\cdot\text{num\_heads}, L, S)` where N is the batch size, L is the target sequence
                length, S is the source sequence length. ``attn_bias`` allows to pass pos embed directly into attention
                If a ByteTensor is provided, the non-zero positions are not allowed to attend while the zero positions will
                be unchanged. If a BoolTensor is provided, positions with ``True`` is not allowed to attend while ``False``
                values will be unchanged. If a FloatTensor is provided, it will be added to the attention weight.
            - attn_mask: if a 2D mask: :math:`(L, S)` where L is the target sequence length, S is the
                source sequence length.
                If a 3D mask: :math:`(N\cdot\text{num\_heads}, L, S)` where N is the batch size, L is the target sequence
                length, S is the source sequence length. ``attn_mask`` ensure that position i is allowed to attend
                the unmasked positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
                while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
                is not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
                is provided, it will be added to the attention weight.
            - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
                If a ByteTensor is provided, the non-zero positions will be ignored while the position
                with the zero positions will be unchanged. If a BoolTensor is provided, the positions with the
                value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
        Outputs:
            - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
                E is the embedding dimension. :math:`(N, L, E)` if ``batch_first`` is ``True``.
            - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
                L is the target sequence length, S is the source sequence length.
        """
        if self.batch_first:
            query, key, value = [x.transpose(0, 1) for x in (query, key, value)]

        # set up shape vars
        target_len, batch_size, embed_dim = query.shape
        source_len, _, _ = key.shape
        if not key.shape[:2] == value.shape[:2]:
            raise ValueError(
                f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
            )

        q, k, v = self.in_projection(query, key, value)

        # prep attention mask
        if attn_mask is not None:
            if attn_mask.dtype == torch.uint8:
                warnings.warn(
                    "attn_mask is of type uint8. This type is deprecated. Please use bool or float tensors instead."
                )
                attn_mask = attn_mask.to(torch.bool)
            elif not (attn_mask.is_floating_point() or attn_mask.dtype == torch.bool):
                raise ValueError(
                    f"attn_mask should have type float or bool, but got {attn_mask.dtype}."
                )
            # ensure attn_mask's dim is 3
            if attn_mask.dim() == 2:
                correct_shape = (target_len, source_len)
                if attn_mask.shape != correct_shape:
                    raise ValueError(
                        f"attn_mask should have shape {correct_shape}, but got {attn_mask.shape}."
                    )
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_shape = (batch_size * self.num_heads, target_len, source_len)
                if attn_mask.shape != correct_shape:
                    raise ValueError(
                        f"attn_mask should have shape {correct_shape}, but got {attn_mask.shape}."
                    )
            else:
                raise RuntimeError(
                    f"attn_mask should have dimension 2 or 3, bug got {attn_mask.dim()}."
                )

        # prep key padding mask
        if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
            warnings.warn(
                "key_padding_mask is of type uint8. This type is deprecated. Please use bool or float tensors instead."
            )
            key_padding_mask = key_padding_mask.to(torch.bool)

        # reshape q, k, v for multihead attention and make em batch first
        q = q.reshape(target_len, batch_size * self.num_heads, self.head_dim).transpose(
            0, 1
        )
        k = k.reshape(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.reshape(-1, batch_size * self.num_heads, self.head_dim).transpose(0, 1)

        # merge key padding and attention masks
        if key_padding_mask is not None:
            if key_padding_mask.shape != (batch_size, source_len):
                raise ValueError(
                    f"key_padding_mask should have shape {(batch_size, source_len)}, but got {key_padding_mask.shape}"
                )
            key_padding_mask = (
                key_padding_mask.view(batch_size, 1, 1, source_len)
                .expand(-1, self.num_heads, -1, -1)
                .reshape(batch_size * self.num_heads, 1, source_len)
            )
            if attn_mask is None:
                attn_mask = key_padding_mask
            elif attn_mask.dtype == torch.bool:
                attn_mask = attn_mask.logical_or(key_padding_mask)
            else:
                attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

        # convert mask to float
        if attn_mask is not None and attn_mask.dtype == torch.bool:
            new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
            new_attn_mask.masked_fill_(attn_mask, float("-inf"))
            attn_mask = new_attn_mask

        # (deep breath) calculate attention and out projection
        attn_output, attn_output_weights = self.attention(q, k, v, attn_bias, attn_mask)
        attn_output = attn_output.transpose(0, 1).reshape(
            target_len, batch_size, embed_dim
        )
        attn_output = self.out_projection(attn_output)

        attn_output_weights = (
            attn_output_weights.view(batch_size, self.num_heads, target_len, source_len)
            if need_weights
            else torch.zeros(0, requires_grad=False)
        )

        if self.batch_first:
            return attn_output.transpose(0, 1), attn_output_weights
        else:
            return attn_output, attn_output_weights

    def in_projection(self, q: Tensor, k: Tensor, v: Tensor) -> List[Tensor]:
        r"""
        Performs the in-projection step of the attention operation, using packed weights.
        Output is a triple containing projection tensors for query, key and value.
        Args:
            q, k, v: query, key and value tensors to be projected. For self-attention,
                these are typically the same tensor; for encoder-decoder attention,
                k and v are typically the same tensor. (We take advantage of these
                identities for performance if they are present.) Regardless, q, k and v
                must share a common embedding dimension; otherwise their shapes may vary.
        Shape:
            Inputs:
            - q: :math:`(..., E)` where E is the embedding dimension
            - k: :math:`(..., E)` where E is the embedding dimension
            - v: :math:`(..., E)` where E is the embedding dimension
            Output:
            - in output list :math:`[q', k', v']`, each output tensor will have the
                same shape as the corresponding input tensor.
        """
        if k is v:
            # self-attention
            if q is k:
                return self.in_proj(q).split(
                    (self.embed_dim, self.k_dim, self.v_dim), dim=-1
                )
            # encoder-decoder attention
            else:
                w_q, w_kv = self.in_proj.weight.split(
                    [self.embed_dim, self.k_dim + self.v_dim]
                )
                b_q, b_kv = (
                    None
                    if self.in_proj.bias is None
                    else self.in_proj.bias.split(
                        [self.embed_dim, self.k_dim + self.v_dim]
                    )
                )
                return (F.linear(q, w_q, b_q),) + F.linear(k, w_kv, b_kv).split(
                    (self.k_dim, self.v_dim), dim=-1
                )
        else:
            w_q, w_k, w_v = self.in_proj.weight.split(
                [self.embed_dim, self.k_dim, self.v_dim]
            )
            b_q, b_k, b_v = (
                None
                if self.in_proj.bias is None
                else self.in_proj.bias.split([self.embed_dim, self.k_dim, self.v_dim])
            )
            return F.linear(q, w_q, b_q), F.linear(k, w_k, b_k), F.linear(v, w_v, b_v)

    def attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_bias: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Computes scaled dot product attention on query, key and value tensors, using
        an optional attention mask if passed, and applying dropout if a probability
        greater than 0.0 is specified.
        Returns a tensor pair containing attended values and attention weights.
        Args:
            q, k, v: query, key and value tensors. See Shape section for shape details.
            attn_mask: optional tensor containing mask values to be added to calculated
                attention. May be 2D or 3D; see Shape section for details.
            attn_bias: optional tensor containing bias values to be added to calculated
                attention. Used for relative positional embedding. May be 2D or 3D; see
                Shape section for details.
        Shape:
            - q: :math:`(B, Nt, E)` where B is batch size, Nt is the target sequence length,
                and E is embedding dimension.
            - key: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
                and E is embedding dimension.
            - value: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
                and E is embedding dimension.
            - attn_bias: either a 3D tensor of shape :math:`(B, Nt, Ns)` or a 2D tensor of
                shape :math:`(Nt, Ns)`.
            - attn_mask: either a 3D tensor of shape :math:`(B, Nt, Ns)` or a 2D tensor of
                shape :math:`(Nt, Ns)`.
            - Output: attention values have shape :math:`(B, Nt, E)`; attention weights
                have shape :math:`(B, Nt, Ns)`
        """
        q *= self.scaling
        # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
        attn = torch.bmm(q, k.transpose(-2, -1))
        if attn_bias is not None:
            attn += attn_bias
        if attn_mask is not None:
            attn += attn_mask
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
        output = torch.bmm(attn, v)
        return output, attn

    def out_projection(self, attn_output: Tensor) -> Tensor:
        return self.out_proj(attn_output)
