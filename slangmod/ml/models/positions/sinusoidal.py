import math
import torch as pt
from swak.pt.types import Module, Tensor, Dtype, Device
from ....config import LiteralDevice


class Sinusoidal(Module):

    def __init__(
            self,
            mod_dim: int,
            context: int,
            device: Device | LiteralDevice = 'cpu',
            dtype: Dtype = pt.float
    ) -> None:
        super().__init__()
        self.mod_dim = mod_dim
        self.context = context
        self.device = pt.device(device)
        self.dtype = dtype
        self.register_buffer('positional_encodings', self._encodings)

    @property
    def _span(self) -> Tensor:
        return pt.arange(
            start=0,
            end=self.mod_dim,
            step=2,
            device=self.device,
            dtype=self.dtype
        )

    @property
    def _divisors(self) -> Tensor:
        return pt.exp(self._span * math.log(self.context) / self.mod_dim)

    @property
    def _positions(self) -> Tensor:
        return pt.arange(
            start=0,
            end=self.context,
            device=self.device,
            dtype=self.dtype
        ).unsqueeze(1)

    @property
    def _arguments(self) -> Tensor:
        return self._positions * self._divisors

    @property
    def _encodings(self) -> Tensor:
        p = pt.empty(
            self.context,
            self.mod_dim,
            device=self.device,
            dtype=self.dtype
        )
        p[:, 0::2] = pt.sin(self._arguments)
        p[:, 1::2] = pt.cos(self._arguments)
        return p.unsqueeze(0)

    def forward(self, src: Tensor) -> Tensor:
        return src + self.positional_encodings[:, :src.shape[-2], :]

    def reset_parameters(self) -> None:
        pass