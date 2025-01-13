import math
from typing import Self
import torch as pt
import torch.nn as ptn
import torch.nn.functional as ptnf
from swak.pt.types import Device, Dtype, Tensor, Module
from swak.pt.misc import Identity
from swak.pt.blocks import Block
from ...config import config, LiteralDevice, Devices
from .positions import src_pos_enc, qk_pos_enc

__all__ = [
    'Attention',
    'attention'
]

from torch.nn import MultiheadAttention

# ToDo: Remove positional encodings on src
class Attention(Module):

    def __init__(
            self,
            mod_dim: int,
            n_heads: int,
            bias: bool = True,
            dropout: float = 0.1,
            src_pos_enc: Block = Identity(),
            qk_pos_enc: Block = Identity(),
            device: Device | Devices | LiteralDevice = 'cpu',
            dtype: Dtype = pt.float
    ) -> None:
        super().__init__()
        self.mod_dim = mod_dim
        self.n_heads = n_heads
        self.bias = bias
        self.dropout = dropout
        self.src_pos_enc = src_pos_enc
        self.qk_pos_enc = qk_pos_enc
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
        return 1.0 / math.sqrt(self.head_dim)

    @property
    def has_pos_enc(self) -> bool:
        return (
            not isinstance(self.src_pos_enc, Identity) or
            not isinstance(self.qk_pos_enc, Identity)
        )

    def forward(
            self,
            src: Tensor,
            mask: Tensor | None = None,
            is_causal: bool=True
    ) -> Tensor:
        query, key, value = self.qkv(self.src_pos_enc(src)).chunk(3, -1)

        query = query.unflatten(-1, self.sizes).transpose(-2, -3)
        key = key.unflatten(-1, self.sizes).transpose(-2, -3)
        value = value.unflatten(-1, self.sizes).transpose(-2, -3)

        attended = ptnf.scaled_dot_product_attention(
            query=self.qk_pos_enc(query),
            key=self.qk_pos_enc(key),
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
        self.src_pos_enc.reset_parameters()
        self.qk_pos_enc.reset_parameters()

    def new(self) -> Self:
        return self.__class__(
            self.mod_dim,
            self.n_heads,
            self.bias,
            self.dropout,
            self.src_pos_enc.new(),
            self.qk_pos_enc.new(),
            self.device,
            self.dtype
        )


attention = Identity() if config.model.reference else Attention(
    mod_dim=config.model.dim,
    n_heads=config.model.n_heads,
    bias=config.model.bias,
    dropout=config.model.dropout,
    src_pos_enc=src_pos_enc,
    qk_pos_enc=qk_pos_enc,
    device=config.data.device,
    dtype=config.data.dtype
)
