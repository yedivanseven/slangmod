from typing import Any, Self
import math
import torch as pt
from swak.pt.types import Tensor, Dtype, Device
from swak.pt.blocks import Block
from ....config import LiteralDevice, Devices


class Sinusoidal(Block):
    """Sinusoidal positional encodings for transformer-based sequence models.

    Parameters
    ----------
    mod_dim: int
        The model dimension. Inputs are expected to be of that size in their
        last dimension.
    context: int
        The maximum sequence length that can be processed. Inputs are
        expected to not exceed this size in their next-to-last dimension.
    device: str or device, optional
        Torch device to create the learnable positional encodings on.
        Defaults to "cpu".
    dtype: dtype, optional
        Torch dtype of the learnable positional encodings.
        Defaults to ``torch.float``.

    """

    def __init__(
            self,
            mod_dim: int,
            context: int,
            device: Device | Devices | LiteralDevice = 'cpu',
            dtype: Dtype = pt.float,
            **_: Any
    ) -> None:
        super().__init__()
        self.mod_dim = mod_dim
        self.context = context
        self.device = pt.device(device)
        self.dtype = dtype
        self.register_buffer('positional_encodings', self._encodings, False)

    @property
    def _span(self) -> Tensor:
        """Even integer numbers across the embedding/model dimension."""
        return pt.arange(
            start=0,
            end=self.mod_dim,
            step=2,
            device=self.device,
            dtype=self.dtype
        )

    @property
    def _divisors(self) -> Tensor:
        """Multiplicative factors of position indices in sin/cos arguments."""
        return pt.exp(-self._span * math.log(self.context) / self.mod_dim)

    @property
    def _positions(self) -> Tensor:
        """Indices of the positions in the sequence."""
        return pt.arange(
            start=0,
            end=self.context,
            device=self.device,
            dtype=self.dtype
        ).unsqueeze(1)

    @property
    def _angles(self) -> Tensor:
        """Arguments of the trigonometric sine and cosine functions."""
        return self._positions * self._divisors

    @property
    def _encodings(self) -> Tensor:
        """Final, additive positional encodings."""
        encodings = pt.empty(
            1,
            self.context,
            self.mod_dim,
            device=self.device,
            dtype=self.dtype
        )
        encodings[:, :, 0::2] = pt.sin(self._angles)
        encodings[:, :, 1::2] = pt.cos(self._angles)
        return encodings

    def forward(self, src: Tensor) -> Tensor:
        """Add sinusoidal positional encodings to a sequence of embeddings.

        Parameters
        ----------
        src: Tensor
            Input sequence(s). Must be of dimensions (..., `S`, `mod_dim`),
            where the sequence length `S` must not exceed `context`.

        Returns
        -------
        Tensor
            The input sequence(s) with sinusoidal positional encodings added.

        """
        return src + self.positional_encodings[:, :src.size(-2), :]

    def reset_parameters(self) -> None:
        """Does nothing because there are no internal parameters to reset."""


    def new(self) -> Self:
        """Return a fresh, new instance with exactly the same parameters."""
        return self.__class__(
            self.mod_dim,
            self.context,
            self.device,
            self.dtype
        )
