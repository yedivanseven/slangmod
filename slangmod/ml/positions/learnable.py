import torch.nn as ptn
from swak.pt import device
from swak.pt.types import Tensor, Device, Dtype
from ...config import config


class Learnable(ptn.Module):

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
        self.positional_encodings = ptn.Embedding(
            num_embeddings=context,
            embedding_dim=mod_dim,
            dtype=dtype,
            device=device
        )

    def forward(self, src: Tensor) -> Tensor:
        return src + self.positional_encodings

    def reset_parameters(self) -> None:
        self.positional_encodings.reset_parameters()


learnable = Learnable(
    config.model.mod_dim,
    config.data.context,
    config.data.dtype,
    device
)
