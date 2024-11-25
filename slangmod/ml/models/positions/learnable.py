import torch as pt
import torch.nn as ptn
from swak.pt.types import Module, Tensor, Dtype, Device
from ....config import config, LiteralDevice


class Learnable(Module):

    def __init__(
            self,
            mod_dim: int,
            context: int,
            device: Device | LiteralDevice,
            dtype: Dtype
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
        return src + self.positional_encodings[:, :src.shape[1], :]

    def reset_parameters(self) -> None:
        ptn.init.normal_(self.positional_encodings)


learnable = Learnable(
    mod_dim=config.model.dim,
    context=config.data.seq_len,
    device=config.data.device,
    dtype=config.data.dtype,
)
