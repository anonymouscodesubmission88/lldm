# Implementation from the official code repo of:
# "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers." by
# Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong, Jayant Kalagnanam, ICLR 2023
#
# Official git repo: https://github.com/yuqinie98/PatchTST/

from typing import Optional, Sequence, Union
from DynamicalSystems.models.fc_models import MCDropout
from DynamicalSystems.utils.defaults import MODELS_TENSOR_PREDICITONS_KEY

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


def get_activation_fn(activation):
    if callable(activation):
        return activation()

    elif activation.lower() == "relu":
        return nn.ReLU()

    elif activation.lower() == "gelu":
        return nn.GELU()

    raise ValueError(
        f'{activation} is not available. You can use "relu", "gelu", or a callable'
    )


def PositionalEncoding(q_len, hidden_size, normalize=True):
    pe = torch.zeros(q_len, hidden_size)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, hidden_size, 2) * -(math.log(10000.0) / hidden_size)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe


SinCosPosEncoding = PositionalEncoding


def Coord2dPosEncoding(q_len, hidden_size, exponential=False, normalize=True, eps=1e-3):
    x = 0.5 if exponential else 1
    for i in range(100):
        cpe = (
                2
                * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x)
                * (torch.linspace(0, 1, hidden_size).reshape(1, -1) ** x)
                - 1
        )
        if abs(cpe.mean()) <= eps:
            break

        elif cpe.mean() > eps:
            x += 0.001

        else:
            x -= 0.001

        i += 1

    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)

    return cpe


def Coord1dPosEncoding(q_len, exponential=False, normalize=True):
    cpe = (
            2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** (0.5 if exponential else 1))
            - 1
    )
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


def positional_encoding(pe, learn_pe, q_len, hidden_size):
    # Positional encoding
    if pe == None:
        W_pos = torch.empty(
            (q_len, hidden_size)
        )  # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == "zero":
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == "zeros":
        W_pos = torch.empty((q_len, hidden_size))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == "normal" or pe == "gauss":
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == "uniform":
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == "lin1d":
        W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
    elif pe == "exp1d":
        W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
    elif pe == "lin2d":
        W_pos = Coord2dPosEncoding(
            q_len, hidden_size, exponential=False, normalize=True
        )
    elif pe == "exp2d":
        W_pos = Coord2dPosEncoding(q_len, hidden_size, exponential=True, normalize=True)
    elif pe == "sincos":
        W_pos = PositionalEncoding(q_len, hidden_size, normalize=True)
    else:
        raise ValueError(
            f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)"
        )
    return nn.Parameter(W_pos, requires_grad=learn_pe)


class RevIN(nn.Module):
    def __init__(
            self,
            num_features: int,
            eps: float = 1e-5,
            affine: bool = True,
            subtract_last: bool = False,
    ):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last

        if self.affine:
            self._init_params()

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)

        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()

        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last

        else:
            x = x - self.mean

        x = x / self.stdev

        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias

        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)

        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last

        else:
            x = x + self.mean

        return x

    def __call__(self, x, mode: str):
        return self.forward(x=x, mode=mode)

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)

        elif mode == "denorm":
            x = self._denormalize(x)

        else:
            raise NotImplementedError

        return x


class PatchTST_backbone(nn.Module):
    def __init__(
            self,
            c_in: int,
            c_out: int,
            input_size: int,
            h: int,
            patch_len: int,
            stride: int,
            n_layers: int = 3,
            hidden_size=128,
            n_heads=16,
            d_k: Optional[int] = None,
            d_v: Optional[int] = None,
            linear_hidden_size: int = 256,
            attn_dropout: float = 0.0,
            dropout: float = 0.0,
            act: str = "gelu",
            res_attention: bool = True,
            pre_norm: bool = False,
            store_attn: bool = False,
            pe: str = "zeros",
            learn_pe: bool = True,
            fc_dropout: float = 0.0,
            head_dropout: float = 0.0,
            padding_patch: str = None,
            pretrain_head: bool = False,
            head_type: str = "flatten",
            individual: bool = False,
            revin: bool = True,
            affine: bool = True,
            subtract_last: bool = False,
            mc_dropout: bool = False,
    ):
        super().__init__()

        self._mc_dropout = mc_dropout

        # RevIn
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        patch_num = int((input_size - patch_len) / stride + 1)
        if padding_patch == "end":  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            patch_num += 1

        # Backbone
        self.backbone = TSTiEncoder(
            c_in,
            patch_len=patch_len,
            n_layers=n_layers,
            hidden_size=hidden_size,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            linear_hidden_size=linear_hidden_size,
            attn_dropout=attn_dropout,
            dropout=dropout,
            act=act,
            res_attention=res_attention,
            pre_norm=pre_norm,
            store_attn=store_attn,
            pe=pe,
            learn_pe=learn_pe,
            mc_dropout=mc_dropout,
        )

        # Head
        self.head_nf = hidden_size * patch_num
        self.n_vars = c_in
        self.c_out = c_out
        self.pretrain_head = pretrain_head
        self.head_type = head_type
        self.individual = individual

        if self.pretrain_head:
            self.head = self.create_pretrain_head(
                self.head_nf, c_in, fc_dropout
            )  # custom head passed as a partial func with all its kwargs

        elif head_type == "flatten":
            self.head = Flatten_Head(
                self.individual,
                self.n_vars,
                self.head_nf,
                h,
                c_out,
                head_dropout=head_dropout,
                mc_dropout=mc_dropout,
            )

    def __call__(self, x: torch.Tensor):
        return self.forward(x)

    def forward(self, z):  # z: [bs x nvars x seq_len]
        # norm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, "norm")
            z = z.permute(0, 2, 1)

        # do patching
        if self.padding_patch == "end":
            z = self.padding_patch_layer(z)

        z = z.unfold(
            dimension=-1, size=self.patch_len, step=self.stride
        )  # z: [bs x nvars x patch_num x patch_len]

        z = z.permute(0, 1, 3, 2)  # z: [bs x nvars x patch_len x patch_num]

        # model
        z = self.backbone(z)  # z: [bs x nvars x hidden_size x patch_num]
        z = self.head(z)  # z: [bs x nvars x h]

        # denorm
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, "denorm")
            z = z.permute(0, 2, 1)

        return z

    def create_pretrain_head(self, head_nf, vars, dropout):
        if self._mc_dropout and dropout > 0:
            dropout_layer = MCDropout(dropout)

        else:
            dropout_layer = nn.Dropout(dropout)

        return nn.Sequential(dropout_layer, nn.Conv1d(head_nf, vars, 1))


class Flatten_Head(nn.Module):
    def __init__(
            self,
            individual,
            n_vars,
            nf,
            h,
            c_out,
            head_dropout: float = 0.0,
            mc_dropout: bool = False,
    ):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars
        self.c_out = c_out
        self._mc_dropout = mc_dropout

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, h * c_out))

                if mc_dropout and head_dropout > 0:
                    self.dropouts.append(MCDropout(head_dropout))

                else:
                    self.dropouts.append(nn.Dropout(head_dropout))

        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, h * c_out)

            if mc_dropout and head_dropout > 0:
                self.dropout = MCDropout(head_dropout)

            else:
                self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x hidden_size x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])  # z: [bs x hidden_size * patch_num]
                z = self.linears[i](z)  # z: [bs x h]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # x: [bs x nvars x h]

        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x
                             )
        return x


class TSTiEncoder(nn.Module):  # i means channel-independent
    def __init__(
            self,
            patch_num: Union[int, Sequence[int]],
            patch_len,
            n_layers=3,
            hidden_size=128,
            n_heads=16,
            d_k=None,
            d_v=None,
            linear_hidden_size=256,
            norm="BatchNorm",
            attn_dropout=0.0,
            dropout=0.0,
            act="gelu",
            store_attn=False,
            res_attention=True,
            pre_norm=False,
            pe="zeros",
            learn_pe=True,
            mc_dropout: bool = False,
    ):
        super().__init__()

        self.patch_num = patch_num
        self.patch_len = patch_len
        self._mc_dropout = mc_dropout

        # Input encoding
        q_len = patch_num
        self.W_P = nn.Linear(
            patch_len, hidden_size
        )  # Eq 1: projection of feature vectors onto a d-dim vector space
        self.seq_len = q_len

        # Positional encoding
        if isinstance(q_len, int):
            self._single_length = True
            self.W_pos = positional_encoding(pe, learn_pe, q_len, hidden_size)

        else:
            self._single_length = False
            self.W_pos_predict = positional_encoding(pe, learn_pe, q_len[0], hidden_size)
            self.W_pos_restore = positional_encoding(pe, learn_pe, q_len[1], hidden_size)

        # Residual dropout
        if mc_dropout and dropout > 0:
            self.dropout = MCDropout(dropout)

        else:
            self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(
            hidden_size,
            n_heads,
            d_k=d_k,
            d_v=d_v,
            linear_hidden_size=linear_hidden_size,
            norm=norm,
            attn_dropout=attn_dropout,
            dropout=dropout,
            pre_norm=pre_norm,
            activation=act,
            res_attention=res_attention,
            n_layers=n_layers,
            store_attn=store_attn,
            mc_dropout=mc_dropout,
        )

    def __call__(self, x) -> torch.Tensor:
        return self.forward(x)

    def forward(self, x) -> torch.Tensor:  # x: [bs x nvars x patch_len x patch_num]
        n_vars = x.shape[1]

        # Input encoding
        x = x.permute(0, 1, 3, 2)  # x: [bs x nvars x patch_num x patch_len]
        x = self.W_P(x)  # x: [bs x nvars x patch_num x hidden_size]
        u = torch.reshape(
            x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3])
        )  # u: [bs * nvars x patch_num x hidden_size]

        if self._single_length:
            u = self.dropout(u + self.W_pos)  # u: [bs * nvars x patch_num x hidden_size]

        else:
            if x.shape[2] == self.patch_num[0]:
                u = self.dropout(u + self.W_pos_predict)  # u: [bs * nvars x patch_num x hidden_size]

            elif x.shape[2] == self.patch_num[1]:
                u = self.dropout(u + self.W_pos_restore)  # u: [bs * nvars x patch_num x hidden_size]

            else:
                raise ValueError(f"Invalid shape for X: {x.shape}")

        # Encoder
        z = self.encoder(u)  # z: [bs * nvars x patch_num x hidden_size]
        z = torch.reshape(
            z, (-1, n_vars, z.shape[-2], z.shape[-1])
        )  # z: [bs x nvars x patch_num x hidden_size]
        z = z.permute(0, 1, 3, 2)  # z: [bs x nvars x hidden_size x patch_num]

        return z


class TSTEncoder(nn.Module):
    def __init__(
            self,
            hidden_size,
            n_heads,
            d_k=None,
            d_v=None,
            linear_hidden_size=None,
            norm="BatchNorm",
            attn_dropout: float = 0.0,
            dropout: float = 0.0,
            activation="gelu",
            res_attention=False,
            n_layers=1,
            pre_norm=False,
            store_attn=False,
            mc_dropout: bool = False,
    ):
        super().__init__()
        self._mc_dropout = mc_dropout
        self.layers = nn.ModuleList(
            [
                TSTEncoderLayer(
                    hidden_size,
                    n_heads=n_heads,
                    d_k=d_k,
                    d_v=d_v,
                    linear_hidden_size=linear_hidden_size,
                    norm=norm,
                    attn_dropout=attn_dropout,
                    dropout=dropout,
                    activation=activation,
                    res_attention=res_attention,
                    pre_norm=pre_norm,
                    store_attn=store_attn,
                    mc_dropout=mc_dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.res_attention = res_attention

    def __call__(
            self,
            src: torch.Tensor,
            key_padding_mask: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        return self.forward(src=src, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

    def forward(
            self,
            src: torch.Tensor,
            key_padding_mask: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        output = src
        scores = None
        if self.res_attention:
            for mod in self.layers:
                output, scores = mod(
                    output,
                    prev=scores,
                    key_padding_mask=key_padding_mask,
                    attn_mask=attn_mask,
                )

            return output

        else:
            for mod in self.layers:
                output = mod(
                    output, key_padding_mask=key_padding_mask, attn_mask=attn_mask
                )

            return output


class TSTEncoderLayer(nn.Module):
    def __init__(
            self,
            hidden_size,
            n_heads,
            d_k=None,
            d_v=None,
            linear_hidden_size=256,
            store_attn=False,
            norm="BatchNorm",
            attn_dropout: float = 0.0,
            dropout: float = 0.0,
            bias=True,
            activation="gelu",
            res_attention=False,
            pre_norm=False,
            mc_dropout: bool = False,
    ):
        super().__init__()
        assert (
            not hidden_size % n_heads
        ), f"hidden_size ({hidden_size}) must be divisible by n_heads ({n_heads})"
        d_k = hidden_size // n_heads if d_k is None else d_k
        d_v = hidden_size // n_heads if d_v is None else d_v

        # Multi-Head attention
        self._mc_dropout = mc_dropout
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(
            hidden_size,
            n_heads,
            d_k,
            d_v,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
            res_attention=res_attention,
        )

        # Add & Norm
        if mc_dropout and dropout > 0:
            self.dropout_attn = MCDropout(dropout)

        else:
            self.dropout_attn = nn.Dropout(dropout)

        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(hidden_size), Transpose(1, 2)
            )

        else:
            self.norm_attn = nn.LayerNorm(hidden_size)

        # Position-wise Feed-Forward
        if mc_dropout and dropout > 0:
            dropout_layer = MCDropout(dropout)

        else:
            dropout_layer = nn.Dropout(dropout)

        self.ff = nn.Sequential(
            nn.Linear(hidden_size, linear_hidden_size, bias=bias),
            get_activation_fn(activation),
            dropout_layer,
            nn.Linear(linear_hidden_size, hidden_size, bias=bias),
        )

        # Add & Norm
        if mc_dropout and dropout > 0:
            self.dropout_ffn = MCDropout(dropout)

        else:
            self.dropout_ffn = nn.Dropout(dropout)

        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(
                Transpose(1, 2), nn.BatchNorm1d(hidden_size), Transpose(1, 2)
            )

        else:
            self.norm_ffn = nn.LayerNorm(hidden_size)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

    def __call__(
            self,
            src: torch.Tensor,
            prev: Optional[torch.Tensor] = None,
            key_padding_mask: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        return self.forward(src=src, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)

    def forward(
            self,
            src: torch.Tensor,
            prev: Optional[torch.Tensor] = None,
            key_padding_mask: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):  # -> Tuple[torch.Tensor, Any]:
        # Multi-Head attention sublayer
        if self.pre_norm:
            src = self.norm_attn(src)

        ## Multi-Head attention
        if self.res_attention:
            src2, attn, scores = self.self_attn(
                src,
                src,
                src,
                prev,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )

        else:
            src2, attn = self.self_attn(
                src, src, src, key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )

        if self.store_attn:
            self.attn = attn

        ## Add & Norm
        src = src + self.dropout_attn(
            src2
        )  # Add: residual connection with residual dropout
        if not self.pre_norm:
            src = self.norm_attn(src)

        # Feed-forward sublayer
        if self.pre_norm:
            src = self.norm_ffn(src)

        ## Position-wise Feed-Forward
        src2 = self.ff(src)

        ## Add & Norm
        src = src + self.dropout_ffn(
            src2
        )  # Add: residual connection with residual dropout

        if not self.pre_norm:
            src = self.norm_ffn(src)

        if self.res_attention:
            return src, scores

        else:
            return src


class _MultiheadAttention(nn.Module):
    def __init__(
            self,
            hidden_size,
            n_heads,
            d_k=None,
            d_v=None,
            res_attention=False,
            attn_dropout=0.0,
            proj_dropout=0.0,
            qkv_bias=True,
            lsa=False,
    ):
        """
        Multi Head Attention Layer
        Input shape:
            Q:       [batch_size (bs) x max_q_len x hidden_size]
            K, V:    [batch_size (bs) x q_len x hidden_size]
            mask:    [q_len x q_len]
        """
        super().__init__()
        d_k = hidden_size // n_heads if d_k is None else d_k
        d_v = hidden_size // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(hidden_size, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(hidden_size, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(hidden_size, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        self.sdp_attn = _ScaledDotProductAttention(
            hidden_size,
            n_heads,
            attn_dropout=attn_dropout,
            res_attention=self.res_attention,
            lsa=lsa,
        )

        # Poject output
        self.to_out = nn.Sequential(
            nn.Linear(n_heads * d_v, hidden_size), nn.Dropout(proj_dropout)
        )

    def forward(
            self,
            Q: torch.Tensor,
            K: Optional[torch.Tensor] = None,
            V: Optional[torch.Tensor] = None,
            prev: Optional[torch.Tensor] = None,
            key_padding_mask: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        bs = Q.size(0)
        if K is None:
            K = Q

        if V is None:
            V = Q

        # Linear (+ split in multiple heads)
        q_s = (
            self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)
        )  # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = (
            self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0, 2, 3, 1)
        )  # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = (
            self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)
        )  # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(
                q_s,
                k_s,
                v_s,
                prev=prev,
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )

        else:
            output, attn_weights = self.sdp_attn(
                q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = (
            output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)
        )  # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention:
            return output, attn_weights, attn_scores

        else:
            return output, attn_weights


class _ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention module (Attention is all you need by Vaswani et al., 2017) with optional residual attention from previous layer
    (Realformer: Transformer likes residual attention by He et al, 2020) and locality self sttention (Vision Transformer for Small-Size Datasets
    by Lee et al, 2021)
    """

    def __init__(
            self, hidden_size, n_heads, attn_dropout=0.0, res_attention=False, lsa=False
    ):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = hidden_size // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            prev: Optional[torch.Tensor] = None,
            key_padding_mask: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        """
        Input shape:
            q               : [bs x n_heads x max_q_len x d_k]
            k               : [bs x n_heads x d_k x seq_len]
            v               : [bs x n_heads x seq_len x d_v]
            prev            : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask       : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        """

        # Scaled MatMul (q, k) - similarity scores for all pairs of positions in an input sequence
        attn_scores = (
                torch.matmul(q, k) * self.scale
        )  # attn_scores : [bs x n_heads x max_q_len x q_len]

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None:
            attn_scores = attn_scores + prev

        # Attention mask (optional)
        if (
                attn_mask is not None
        ):  # attn_mask with shape [q_len x seq_len] - only used when q_len == seq_len
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if (
                key_padding_mask is not None
        ):  # mask with shape [bs x q_len] (only when max_w_len == q_len)
            attn_scores.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf
            )

        # normalize the attention weights
        attn_weights = F.softmax(
            attn_scores, dim=-1
        )  # attn_weights   : [bs x n_heads x max_q_len x q_len]
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(
            attn_weights, v
        )  # output: [bs x n_heads x max_q_len x d_v]

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights


class PatchTST(nn.Module):
    """PatchTST

    The PatchTST model is an efficient Transformer-based model for multivariate time series forecasting.

    It is based on two key components:
    - segmentation of time series into windows (patches) which are served as input tokens to Transformer
    - channel-independence, where each channel contains a single univariate time series.

    **Parameters:**<br>
    `h`: int, Forecast horizon. <br>
    `input_size`: int, autorregresive inputs size, y=[1,2,3,4] input_size=2 -> y_[t-2:t]=[1,2].<br>
    `stat_exog_list`: str list, static exogenous columns.<br>
    `hist_exog_list`: str list, historic exogenous columns.<br>
    `futr_exog_list`: str list, future exogenous columns.<br>
    `exclude_insample_y`: bool=False, the model skips the autoregressive features y[t-input_size:t] if True.<br>
    `encoder_layers`: int, number of layers for encoder.<br>
    `n_heads`: int=16, number of multi-head's attention.<br>
    `hidden_size`: int=128, units of embeddings and encoders.<br>
    `linear_hidden_size`: int=256, units of linear layer.<br>
    `dropout`: float=0.1, dropout rate for residual connection.<br>
    `fc_dropout`: float=0.1, dropout rate for linear layer.<br>
    `head_dropout`: float=0.1, dropout rate for Flatten head layer.<br>
    `attn_dropout`: float=0.1, dropout rate for attention layer.<br>
    `patch_len`: int=32, length of patch.<br>
    `stride`: int=16, stride of patch.<br>
    `revin`: bool=True, bool to use RevIn.<br>
    `revin_affine`: bool=False, bool to use affine in RevIn.<br>
    `revin_substract_last`: bool=False, bool to use substract last in RevIn.<br>
    `activation`: str='ReLU', activation from ['gelu','relu'].<br>
    `res_attention`: bool=False, bool to use residual attention.<br>
    `batch_normalization`: bool=False, bool to use batch normalization.<br>
    `learn_pos_embedding`: bool=True, bool to learn positional embedding.<br>
    `loss`: PyTorch module, instantiated train loss class from [losses collection](https://nixtla.github.io/neuralforecast/losses.pytorch.html).<br>
    `valid_loss`: PyTorch module=`loss`, instantiated valid loss class from [losses collection](https://nixtla.github.io/neuralforecast/losses.pytorch.html).<br>
    `max_steps`: int=1000, maximum number of training steps.<br>
    `learning_rate`: float=1e-3, Learning rate between (0, 1).<br>
    `num_lr_decays`: int=-1, Number of learning rate decays, evenly distributed across max_steps.<br>
    `early_stop_patience_steps`: int=-1, Number of validation iterations before early stopping.<br>
    `val_check_steps`: int=100, Number of training steps between every validation loss check.<br>
    `batch_size`: int=32, number of different series in each batch.<br>
    `valid_batch_size`: int=None, number of different series in each validation and test batch, if None uses batch_size.<br>
    `windows_batch_size`: int=1024, number of windows to sample in each training batch, default uses all.<br>
    `inference_windows_batch_size`: int=1024, number of windows to sample in each inference batch.<br>
    `step_size`: int=1, step size between each window of temporal data.<br>
    `scaler_type`: str='identity', type of scaler for temporal inputs normalization see [temporal scalers](https://nixtla.github.io/neuralforecast/common.scalers.html).<br>
    `random_seed`: int, random_seed for pytorch initializer and numpy generators.<br>
    `num_workers_loader`: int=os.cpu_count(), workers to be used by `TimeSeriesDataLoader`.<br>
    `drop_last_loader`: bool=False, if True `TimeSeriesDataLoader` drops last non-full batch.<br>
    `alias`: str, optional,  Custom name of the model.<br>
    `**trainer_kwargs`: int,  keyword trainer arguments inherited from [PyTorch Lighning's trainer](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.trainer.trainer.Trainer.html?highlight=trainer).<br>

    **References:**<br>
    -[Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2022). "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"](https://arxiv.org/pdf/2211.14730.pdf)
    """

    def __init__(
            self,
            horizon: int,
            input_size: int,
            encoder_layers: int = 3,
            n_heads: int = 16,
            hidden_size: int = 128,
            linear_hidden_size: int = 256,
            dropout: float = 0.2,
            fc_dropout: float = 0.2,
            head_dropout: float = 0.0,
            attn_dropout: float = 0.0,
            patch_len: int = 16,
            stride: int = 8,
            revin: bool = True,
            revin_affine: bool = False,
            revin_subtract_last: bool = True,
            activation: str = "gelu",
            res_attention: bool = True,
            batch_normalization: bool = False,
            learn_pos_embed: bool = True,
            mc_dropout: bool = False,
            n_classes: Optional[int] = None,
    ):
        super().__init__()

        # Fixed hyperparameters
        c_in = 1  # Always univariate
        padding_patch = "end"  # Padding at the end
        pretrain_head = False  # No pretrained head
        pe = "zeros"  # Initial zeros for positional encoding
        d_k = None  # Key dimension
        d_v = None  # Value dimension
        store_attn = False  # Store attention weights
        head_type = "flatten"  # Head type
        individual = False  # Separate heads for each time series

        self.model = PatchTST_backbone(
            c_in=c_in,
            c_out=c_in,
            input_size=input_size,
            h=horizon,
            patch_len=patch_len,
            stride=stride,
            n_layers=encoder_layers,
            hidden_size=hidden_size,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            linear_hidden_size=linear_hidden_size,
            attn_dropout=attn_dropout,
            dropout=dropout,
            act=activation,
            res_attention=res_attention,
            pre_norm=batch_normalization,
            store_attn=store_attn,
            pe=pe,
            learn_pe=learn_pos_embed,
            fc_dropout=fc_dropout,
            head_dropout=head_dropout,
            padding_patch=padding_patch,
            pretrain_head=pretrain_head,
            head_type=head_type,
            individual=individual,
            revin=revin,
            affine=revin_affine,
            subtract_last=revin_subtract_last,
            mc_dropout=mc_dropout,
        )

        self._n_classes = n_classes
        self._classify = False
        if n_classes is not None:
            self._classify = True
            self._classifier = torch.nn.Sequential(
                torch.nn.Linear(horizon, 512),
                torch.nn.LeakyReLU(0.1),
                torch.nn.Linear(512, 1024),
                torch.nn.LeakyReLU(0.1),
                torch.nn.Linear(1024, (horizon * n_classes)),
            )

    def __call__(self, x: torch.Tensor):
        return self.forward(x)

    def forward(self, x: torch.Tensor):  # x: [batch, channels, 1, input_size]
        if len(x.shape) == 4:
            out = self.model(x[:, :, 0, :])[..., None, :]

        else:
            x = x.permute(0, 2, 1)
            out = self.model(x)
            out = out.permute(0, 2, 1)

        if self._classify:
            out = out.squeeze()
            assert len(out.shape) == 2
            out = self._classifier(out)
            out = out.reshape(-1, self._n_classes)

        out = {
            MODELS_TENSOR_PREDICITONS_KEY: out,
        }

        return out
