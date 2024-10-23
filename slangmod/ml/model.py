import torch as pt
import torch.nn as ptn
from swak.funcflow import Partial
from swak.pt import device
from swak.pt.types import Module, Tensor, Tensors1T, Dtype, Device
from .positions import positions
from ..config import config


class Model(ptn.Module):

    def __init__(
            self,
            mod_dim: int,
            context: int,
            vocab_size: int,
            positions: Module,
            n_heads: int,
            n_layers: int,
            scale_grad_by_freq: bool,
            dropout: float,
            bias: bool,
            dtype: Dtype,
            device: Device
    ) -> None:
        super().__init__()
        self.mod_dim = mod_dim
        self.context = context
        self.vocab_size = vocab_size
        self.positions = positions
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.scale_grad_by_freq = scale_grad_by_freq
        self.dropout = dropout
        self.bias = bias
        self.dtype = dtype
        self.device = device
        self.embed = ptn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=mod_dim,
            padding_idx=0,
            scale_grad_by_freq=scale_grad_by_freq,
            dtype=dtype,
            device=device,
        )
        self.encoder = ptn.TransformerEncoderLayer(
            d_model=mod_dim,
            nhead=n_heads,
            dim_feedforward=4 * mod_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False,
            bias=bias,
            dtype=dtype,
            device=device,
        )
        self.transform = ptn.TransformerEncoder(
            encoder_layer=self.encoder,
            num_layers=n_layers,
            enable_nested_tensor=False,
            mask_check=False
        )
        self.finalize = ptn.Linear(
            in_features=mod_dim,
            out_features=vocab_size,
            bias=bias,
            dtype=dtype,
            device=device,
        )

    def forward(
            self,
            src: Tensor,
            mask: Tensor | None,
            padding_mask: Tensor | None,
            is_causal: bool
    ) -> Tensors1T:
        embedded = self.positions(self.embed(src))
        transformed = self.transform(embedded, mask, padding_mask, is_causal)
        return self.finalize(transformed).transpose(-1, -2).contiguous(),

    def reset_parameters(self) -> None:
        self.embed.reset_parameters()
        self.positions.reset_parameters()
        self.finalize.reset_parameters()
        self.encoder = ptn.TransformerEncoderLayer(
            d_model=self.mod_dim,
            nhead=self.n_heads,
            dim_feedforward=4 * self.mod_dim,
            dropout=self.dropout,
            batch_first=True,
            norm_first=False,
            bias=self.bias,
            dtype=self.dtype,
            device=self.device,
        )
        self.transform = ptn.TransformerEncoder(
            encoder_layer=self.encoder,
            num_layers=self.n_layers,
            enable_nested_tensor=False,
            mask_check=False
        )


model = Model(
    config.model.mod_dim,
    config.data.context,
    config.tokenizer.vocab_size,
    positions,
    config.model.n_heads,
    config.model.n_layers,
    config.model.scale_grad_by_freq,
    config.model.dropout,
    config.model.bias,
    config.data.dtype,
    device
)

compiled_model = pt.compile(model)
compile_model = Partial[Model](pt.compile, model)
