import math
import torch as pt
from swak.pt.types import Tensor, Dtype
from ..types import Device
from ...config import config


class Sinusoidal(pt.nn.Module):

    def __init__(
            self,
            mod_dim: int,
            max_len: int,
            device: Device,
            dtype: Dtype
    ) -> None:
        super().__init__()
        self.mod_dim = mod_dim
        self.max_len = max_len
        self.device = device
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
        return pt.exp(-2 * self._span * math.log(self.max_len) / self.mod_dim)

    @property
    def _positions(self) -> Tensor:
        return pt.arange(
            start=0,
            end=self.max_len,
            device=self.device,
            dtype=self.dtype
        ).unsqueeze(1)

    @property
    def _arguments(self) -> Tensor:
        return self._positions * self._divisors

    @property
    def _encodings(self) -> Tensor:
        p = pt.empty(
            self.max_len,
            self.mod_dim,
            device=self.device,
            dtype=self.dtype
        )
        p[:, 0::2] = pt.sin(self._arguments)
        p[:, 1::2] = pt.cos(self._arguments)
        return p.unsqueeze(0)

    def forward(self, src: Tensor) -> Tensor:
        return src + self.positional_encodings[:, :src.shape[1], :]

    def reset_parameters(self) -> None:
        pass


sinusoidal = Sinusoidal(
    mod_dim=config.model.dim,
    max_len=config.model.max_len,
    device=config.data.device,
    dtype=config.data.dtype
)
