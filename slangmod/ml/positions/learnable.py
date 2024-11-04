import torch as pt
import torch.nn as ptn
from swak.pt.types import Tensor, Dtype
from ..types import Device
from ...config import config


class Learnable(ptn.Module):

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
        self.positional_encodings = ptn.Parameter(
            pt.empty(1, max_len, mod_dim, device=device, dtype=dtype)
        )

    def forward(self, src: Tensor) -> Tensor:
        return src + self.positional_encodings[:, :src.shape[1], :]

    def reset_parameters(self) -> None:
        ptn.init.normal_(self.positional_encodings)


learnable = Learnable(
    mod_dim=config.model.dim,
    max_len=config.model.max_len,
    device=config.data.device,
    dtype=config.data.dtype,
)
