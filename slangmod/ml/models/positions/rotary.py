from typing import Any
import math
import torch as pt
from swak.pt.types import Module, Tensor, Dtype, Device
from slangmod.config import LiteralDevice


class Rotary(Module):

    def __init__(
            self,
            mod_dim: int,
            n_heads: int,
            context: int,
            device: Device | LiteralDevice = 'cpu',
            dtype: Dtype = pt.float,
            **__: Any
    ) -> None:
        super().__init__()
        self.mod_dim = mod_dim
        self.n_heads = n_heads
        self.context = context
        self.device = pt.device(device)
        self.dtype = dtype
        self.register_buffer('positional_encodings', self._encodings)

    @property
    def head_dim(self) -> int:
        return self.mod_dim // self.n_heads

    @property
    def sizes(self) -> tuple[int, int, int]:
        return self.n_heads, self.head_dim // 2, 2

    @property
    def _span(self) -> Tensor:
        return pt.arange(
            start=0,
            end=self.head_dim,
            step=2,
            device=self.device,
            dtype=self.dtype
        )

    @property
    def _divisors(self) -> Tensor:
        return pt.exp(-self._span * math.log(self.context) / self.head_dim)

    @property
    def _positions(self) -> Tensor:
        return pt.arange(
            start=0,
            end=self.context,
            device=self.device,
            dtype=self.dtype
        ).unsqueeze(1)

    @property
    def angles(self) -> Tensor:
        return self._positions * self._divisors

    @property
    def _encodings(self) -> Tensor:
        encodings = pt.empty(
            1,
            1,
            self.context,
            self.head_dim // 2,
            2,
            device=self.device,
            dtype=self.dtype
        )
        encodings[..., 0] = pt.cos(self.angles)
        encodings[..., 1] = pt.sin(self.angles)
        return encodings

    def forward(self, src: Tensor) -> Tensor:
        seq_len = src.size(-2)
        reshaped = src.unflatten(-1, (-1, 2))
        cos = self.positional_encodings[:, :, :seq_len, :, 0]
        sin = self.positional_encodings[:, :, :seq_len, :, 1]
        return pt.stack([
            reshaped[..., 0] * cos - reshaped[..., 1] * sin,
            reshaped[..., 0] * sin + reshaped[..., 1] * cos
        ], dim=-1).flatten(-2)

    def reset_parameters(self) -> None:
        pass
