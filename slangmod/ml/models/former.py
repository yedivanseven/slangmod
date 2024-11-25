import torch as pt
import torch.nn as ptn
from swak.pt.types import Module, Device, Dtype, Tensor, Tensors1T
from swak.pt.blocks import Block
from swak.pt.misc import Identity
from .layer import vanilla_layer
from .positions import positions
from ...config import config, LiteralDevice

__all__ = [
    'Former',
    'vanilla_former'
]


class Former(Module):

    def __init__(
            self,
            mod_dim: int,
            vocab: int,
            layer: Block,
            n_layers: int,
            positions: Module = Identity,
            bias: bool = True,
            dropout: float = 0.1,
            scale_grad_by_freq: bool = True,
            device: Device | LiteralDevice = 'cpu',
            dtype: Dtype = pt.float
    ) -> None:
        super().__init__()
        self.mod_dim = mod_dim
        self.vocab = vocab
        self.n_layers = n_layers
        self.positions = positions
        self.bias = bias
        self.dropout = dropout
        self.scale_grad_by_freq = scale_grad_by_freq
        self.device = pt.device(device)
        self.dtype = dtype
        self.embed = ptn.Embedding(
            num_embeddings=vocab,
            embedding_dim=mod_dim,
            padding_idx=0,
            scale_grad_by_freq=scale_grad_by_freq,
            device=device,
            dtype=dtype
        )
        self.transform = ptn.Sequential(*[layer.new() for _ in self.layers])
        self.drop = ptn.Dropout(dropout)
        self.finalize = ptn.Linear(
            in_features=mod_dim,
            out_features=vocab,
            bias=bias,
            device=device,
            dtype=dtype
        )

    @property
    def layers(self) -> range:
        return range(self.n_layers)

    def forward(
            self,
            src: Tensor,
            mask: Tensor | None,
            padding_mask: Tensor | None,
            is_causal: bool
    ) -> Tensors1T:
        embedded = self.drop(self.positions(self.embed(src)))
        transformed = self.transform(embedded, mask, padding_mask, is_causal)
        return self.finalize(transformed).contiguous(),

    def reset_parameters(self) -> None:
        for layer in self.transform:
            layer.reset_parameters()
        self.embed.reset_parameters()
        self.positions.reset_parameters()
        self.finalize.reset_parameters()


vanilla_former = Former(
    mod_dim=config.model.dim,
    vocab=config.tokens.vocab,
    layer=vanilla_layer,
    n_layers=config.model.n_layers,
    positions = positions,
    bias = config.model.bias,
    dropout=config.model.bias,
    scale_grad_by_freq=config.model.scale_grad_by_freq,
    device = config.data.device,
    dtype=config.data.device
)
