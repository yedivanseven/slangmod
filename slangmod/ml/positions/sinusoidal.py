import math
import torch as pt
from swak.pt import device
from swak.pt.types import Tensor, Device, Dtype
from ...config import config


class Sinusoidal(pt.nn.Module):

    def __init__(
            self,
            mod_dim: int,
            context: int,
            dtype: Dtype,
            device: Device,
    ) -> None:
        super().__init__()
        self.mod_dim = mod_dim
        self.context = context
        self.dtype = dtype
        self.device = device
        self.register_buffer('positional_encodings', self._encodings)

    @property
    def _span(self) -> Tensor:
        return pt.arange(
            start=0,
            end=self.mod_dim,
            step=2,
            dtype=self.dtype,
            device=self.device
        )

    @property
    def _divisors(self) -> Tensor:
        return pt.exp(-2 * self._span * math.log(self.context) / self.mod_dim)

    @property
    def _positions(self) -> Tensor:
        return pt.arange(
            start=0,
            end=self.context,
            dtype=self.dtype,
            device=self.device
        ).unsqueeze(1)

    @property
    def _arguments(self) -> Tensor:
        return self._positions * self._divisors

    @property
    def _encodings(self) -> Tensor:
        p = pt.empty(
            self.context,
            self.mod_dim,
            dtype=self.dtype,
            device=self.device
        )
        p[:, 0::2] = pt.sin(self._arguments)
        p[:, 1::2] = pt.cos(self._arguments)
        return p.unsqueeze(0)

    def forward(self, src: Tensor) -> Tensor:
        return src + self.positional_encodings

    def reset_parameters(self) -> None:
        pass


sinusoidal = Sinusoidal(
    config.model.mod_dim,
    config.data.context,
    config.data.dtype,
    device
)
