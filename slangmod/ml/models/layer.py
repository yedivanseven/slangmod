from typing import Self
import torch as pt
import torch.nn as ptn
from swak.pt.types import Device, Dtype, Tensor
from swak.pt.blocks import Block
from ...config import config, LiteralDevice
from .attention import vanilla_attention
from .feedforward import vanilla_feedforward


__all__ = [
    'Layer',
    'vanilla_layer'
]


class Layer(Block):

    def __init__(
            self,
            attention: Block,
            feedforward: Block,
            bias: bool = True,
            dropout: float = 0.1,
            norm_first: bool = True,
            eps: float = 1e-5,
            device: Device | LiteralDevice = 'cpu',
            dtype: Dtype = pt.float
    ) -> None:
        super().__init__()
        self.attention = attention
        self.feedforward = feedforward
        self.bias = bias
        self.dropout = dropout
        self.norm_first = norm_first
        self.eps = eps
        self.device = pt.device(device)
        self.dtype = dtype
        self.norm1 = ptn.LayerNorm(
            self.mod_dim,
            eps=eps,
            elementwise_affine=True,
            bias=bias,
            device=device,
            dtype=dtype
        )
        self.norm2 = ptn.LayerNorm(
            self.mod_dim,
            eps=eps,
            elementwise_affine=True,
            bias=bias,
            device=device,
            dtype=dtype
        )
        self.drop1 = ptn.Dropout(dropout)
        self.drop2 = ptn.Dropout(dropout)

    @property
    def mod_dim(self) -> int:
        return self.attention.mod_dim

    def forward(
            self,
            src: Tensor,
            mask: Tensor | None = None,
            is_causal: bool = False,
    ) -> Tensor:
        if self.norm_first:
            attended = self.attention(self.norm1(src), mask, is_causal)
            normed = self.norm2(src + self.drop1(attended))
            out = normed + self.drop2(self.feedforward(normed))
        else:
            attended = self.attention(src, mask, is_causal)
            normed = self.norm1(src + self.drop1(attended))
            out = self.norm2(normed + self.drop2(self.feedforward(normed)))
        return out

    def reset_parameters(self) -> None:
        self.attention.reset_parameters()
        self.feedforward.reset_parameters()
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()

    def new(self) -> Self:
        return self.__class__(
            self.attention.new(),
            self.feedforward.new(),
            self.bias,
            self.dropout,
            self.norm_first,
            self.eps,
            self.device,
            self.dtype,
        )


vanilla_layer = Layer(
    attention=vanilla_attention,
    feedforward=vanilla_feedforward,
    bias=config.model.bias,
    dropout=config.model.dropout,
    norm_first=config.model.norm_first,
    device=config.data.device,
    dtype=config.data.dtype
)
