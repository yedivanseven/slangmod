from typing import Any
import torch as pt
import torch.nn as ptn
from swak.pt.types import Module, Tensor, Dtype, Device
from ....config import LiteralDevice, Devices


class Learnable(Module):

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
        self.positional_encodings = ptn.Parameter(
            pt.empty(1, context, mod_dim, device=device, dtype=dtype)
        )

    def forward(self, src: Tensor) -> Tensor:
        return src + self.positional_encodings[:, :src.size(-2), :]

    def reset_parameters(self) -> None:
        ptn.init.normal_(self.positional_encodings)
