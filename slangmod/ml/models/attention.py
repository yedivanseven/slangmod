import math
from typing import Self
import torch as pt
import torch.nn as ptn
import torch.nn.functional as ptnf
from swak.pt.types import Module, Device, Dtype, Tensor
from ...config import LiteralDevice

__all__ = ['MHA']


class MHA(Module):

    def __init__(
            self,
            mod_dim: int = 512,
            n_heads: int = 8,
            dropout: float = 0.1,
            bias: bool = True,
            device: Device | LiteralDevice = 'cpu',
            dtype: Dtype = pt.float
    ) -> None:
        super().__init__()
        self.mod_dim = mod_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.bias = bias
        self.device = pt.device(device)
        self.dtype = dtype
        self.qkv = ptn.Linear(
            mod_dim,
            3 * mod_dim,
            bias=bias,
            device=device,
            dtype=dtype
        )
        self.out = ptn.Linear(
            mod_dim,
            mod_dim,
            bias=bias,
            device=device,
            dtype=dtype
        )

    @property
    def head_dim(self) -> int:
        return self.mod_dim // self.n_heads

    @property
    def sizes(self) -> tuple[int, int]:
        return self.n_heads, self.head_dim

    @property
    def scale(self) -> float:
        return 1.0 / math.sqrt(self.mod_dim)

    def forward(
            self,
            src: Tensor,
            mask: Tensor | None = None,
            is_causal: bool=True
    ) -> Tensor:
        query, key, value = self.qkv(src).chunk(3, -1)

        query = query.unflatten(-1, self.sizes).transpose(-2, -3)
        key = key.unflatten(-1, self.sizes).transpose(-2, -3)
        value = value.unflatten(-1, self.sizes).transpose(-2, -3)

        attended = ptnf.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=None if is_causal else mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
            scale=self.scale
        )

        return self.out(attended.transpose(-2, -3).flatten(-2))

    def reset_parameters(self) -> None:
        self.qkv.reset_parameters()
        self.out.reset_parameters()

    def new(self) -> Self:
        return self.__class__(
            self.mod_dim,
            self.n_heads,
            self.dropout,
            self.bias,
            self.device,
            self.dtype
        )
