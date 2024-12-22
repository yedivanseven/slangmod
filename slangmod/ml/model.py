import torch as pt
import torch.nn as ptn
from swak.pt.misc import Compile
from swak.pt.types import Module, Tensor, Tensors1T, Dtype, Device
from .models import positions
from ..config import config, LiteralDevice


class Model(Module):

    def __init__(
            self,
            mod_dim: int,
            vocab: int,
            n_heads: int,
            n_layers: int,
            pos_enc: Module,
            feedforward_factor: int,
            scale_grad_by_freq: bool,
            dropout: float,
            bias: bool,
            device: Device | LiteralDevice,
            dtype: Dtype
    ) -> None:
        super().__init__()
        self.mod_dim = mod_dim
        self.vocab = vocab
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.pos_enc = pos_enc
        self.feedforward_factor = feedforward_factor
        self.scale_grad_by_freq = scale_grad_by_freq
        self.dropout = dropout
        self.bias = bias
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
        self.drop = ptn.Dropout(dropout)
        self.encoder = ptn.TransformerEncoderLayer(
            d_model=mod_dim,
            nhead=n_heads,
            dim_feedforward=feedforward_factor * mod_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False,
            bias=bias,
            device=device,
            dtype=dtype
        )
        self.transform = ptn.TransformerEncoder(
            encoder_layer=self.encoder,
            num_layers=n_layers,
            enable_nested_tensor=False,
            mask_check=False
        )
        self.finalize = ptn.Linear(
            in_features=mod_dim,
            out_features=vocab,
            bias=bias,
            device=device,
            dtype=dtype
        )

    def forward(
            self,
            src: Tensor,
            mask: Tensor | None,
            padding_mask: Tensor | None,
            is_causal: bool
    ) -> Tensors1T:
        print(src.shape)
        embedded = self.drop(self.pos_enc(self.embed(src)))
        transformed = self.transform(embedded, mask, padding_mask, is_causal)
        return self.finalize(transformed).transpose(-1, -2).contiguous(),

    def reset_parameters(self) -> None:
        self.embed.reset_parameters()
        self.pos_enc.reset_parameters()
        self.finalize.reset_parameters()
        self.encoder = ptn.TransformerEncoderLayer(
            d_model=self.mod_dim,
            nhead=self.n_heads,
            dim_feedforward=self.feedforward_factor * self.mod_dim,
            dropout=self.dropout,
            batch_first=True,
            norm_first=False,
            bias=self.bias,
            device=self.device,
            dtype=self.dtype
        )
        self.transform = ptn.TransformerEncoder(
            encoder_layer=self.encoder,
            num_layers=self.n_layers,
            enable_nested_tensor=False,
            mask_check=False
        )


model = Model(
    mod_dim=config.model.dim,
    vocab=config.tokens.vocab,
    n_heads=config.model.n_heads,
    n_layers=config.model.n_layers,
    pos_enc=positions,
    feedforward_factor=config.model.feedforward_factor,
    scale_grad_by_freq=config.model.scale_grad_by_freq,
    dropout=config.model.dropout,
    bias=config.model.bias,
    device=config.data.device,
    dtype=config.data.dtype
)

compile_model = Compile(
    inplace=True,
    model=model,
    disable=config.model.disable
)
