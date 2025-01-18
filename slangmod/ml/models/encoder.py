import torch as pt
import torch.nn as ptn
from swak.pt.types import Module, Device, Dtype, Tensor, Tensors1T
from swak.pt.misc import Identity
from .layer import EncoderLayer
from ...config import LiteralDevice, Devices

__all__ = ['Encoder']


class Encoder(Module):

    def __init__(
            self,
            vocab: int,
            pad_id: int,
            n_layers: int,
            layer: EncoderLayer,
            pos_enc: Module = Identity(),
            bias: bool = True,
            dropout: float = 0.1,
            scale_grad_by_freq: bool = True,
            device: Device | Devices | LiteralDevice = 'cpu',
            dtype: Dtype = pt.float
    ) -> None:
        super().__init__()
        self.vocab = vocab
        self.pad_id = pad_id
        self.n_layers = max(1, n_layers)
        self.layers = ptn.ModuleList([
            layer.new().to(device=device, dtype=dtype)
            for _ in range(self.n_layers)
        ])
        self.n_layers = n_layers
        self.pos_enc = pos_enc.to(device=device, dtype=dtype)
        self.bias = bias
        self.dropout = dropout
        self.scale_grad_by_freq = scale_grad_by_freq
        self.embed = ptn.Embedding(
            num_embeddings=vocab,
            embedding_dim=self.mod_dim,
            padding_idx=pad_id,
            scale_grad_by_freq=scale_grad_by_freq,
            device=device,
            dtype=dtype
        )
        self.drop = ptn.Dropout(dropout)
        self.norm = ptn.LayerNorm(
            self.mod_dim,
            eps=layer.norm1.eps,
            elementwise_affine=layer.norm1.elementwise_affine,
            bias=layer.bias,
            device=device,
            dtype=dtype
        ) if layer.norm_first else Identity()
        self.finalize = ptn.Linear(
            in_features=self.mod_dim,
            out_features=vocab,
            bias=bias,
            device=device,
            dtype=dtype
        )

    @property
    def mod_dim(self) -> int:
        """The model dimension."""
        return self.layers[0].mod_dim

    @property
    def device(self) -> Device:
        """The device all weights, biases, activations, etc. reside on."""
        return self.finalize.weight.device

    @property
    def dtype(self) -> Dtype:
        """The dtype of all weights, biases, activations, and parameters."""
        return self.finalize.weight.dtype

    @property
    def context(self) -> int:  # ToDo: Unit test this!
        if hasattr(self.pos_enc, 'context'):
            return min(self.pos_enc.context, self.layer.context)
        return self.layer.context

    def forward(
            self,
            src: Tensor,
            mask: Tensor | None,
            padding_mask: Tensor | None,
            is_causal: bool
    ) -> Tensors1T:
        if is_causal or (mask is None and padding_mask is None):
            attn_mask = None
        elif mask is None:
            sizes = -1, padding_mask.size(-1), -1
            attn_mask = padding_mask.unsqueeze(-2).expand(*sizes)
        elif padding_mask is None:
            attn_mask = mask
        else:
            sizes = -1, padding_mask.size(-1), -1
            attn_mask = mask + padding_mask.unsqueeze(-2).expand(*sizes)

        out = self.drop(self.pos_enc(self.embed(src)))
        for layer in self.layers:
            out = layer(out, attn_mask, is_causal)

        return self.finalize(self.norm(out)).transpose(-1, -2).contiguous(),

    def reset_parameters(self) -> None:
        for layer in self.layers:
            layer.reset_parameters()
        self.embed.reset_parameters()
        self.pos_enc.reset_parameters()
        self.finalize.reset_parameters()
