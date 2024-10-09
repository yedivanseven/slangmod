import math
from functools import cached_property
import torch as pt
from swak.pt.types import Tensor, Device, Dtype
from .abc import Positional


class Sinusoidal(Positional):

    def __init__(
            self,
            mod_dim: int,
            context: int,
            dtype: Dtype,
            device: Device,
    ) -> None:
        self.mod_dim = mod_dim
        self.context = context
        self.dtype = dtype
        self.device = device

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        args = f'{self.mod_dim}, {self.context}, {self.dtype}, {self.device}'
        return f'{cls}({args})'

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

    @cached_property
    def encodings(self) -> Tensor:
        p = pt.empty(
            self.context,
            self.mod_dim,
            dtype=self.dtype,
            device=self.device
        )
        p[:, 0::2] = pt.sin(self._arguments)
        p[:, 1::2] = pt.cos(self._arguments)
        return p.unsqueeze(0)
