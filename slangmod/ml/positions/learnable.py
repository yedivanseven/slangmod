import torch as pt
import torch.nn as ptn
from swak.pt.types import Tensor, Dtype
from ..types import Device
from ...config import config


class Learnable(ptn.Module):

    def __init__(
            self,
            mod_dim: int,
            context: int,
            device: Device,
            dtype: Dtype
    ) -> None:
        super().__init__()
        self.mod_dim = mod_dim
        self.context = context
        self.device = device
        self.dtype = dtype
        self.register_buffer(
            'indices',
            pt.arange(context, device=device, dtype=pt.long)
        )
        self.positional_encodings = ptn.Embedding(
            num_embeddings=context,
            embedding_dim=mod_dim,
            device=device,
            dtype=dtype
        )

    def forward(self, src: Tensor) -> Tensor:
        return src + self.positional_encodings(self.indices)

    def reset_parameters(self) -> None:
        self.positional_encodings.reset_parameters()


learnable = Learnable(
    mod_dim=config.model.mod_dim,
    context=config.data.context,
    device=config.data.device,
    dtype=config.data.dtype,
)
