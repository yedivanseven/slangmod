import torch as pt
import torch.nn as ptn
from swak.pt.types import Module, Device, Dtype, Tensor, Tensors1T
from swak.pt.misc import Identity
from .layer import Layer
from ...config import LiteralDevice, Devices

__all__ = ['Former']

class Former(Module):

    def __init__(
            self,
            mod_dim: int,
            vocab: int,
            layer: Layer,
            n_layers: int,
            emb_pos_enc: Module = Identity(),
            bias: bool = True,
            dropout: float = 0.1,
            scale_grad_by_freq: bool = True,
            device: Device | Devices | LiteralDevice = 'cpu',
            dtype: Dtype = pt.float
    ) -> None:
        super().__init__()
        self.mod_dim = mod_dim
        self.vocab = vocab
        self.n_layers = n_layers
        self.emb_pos_enc = emb_pos_enc
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
        self.drop = ptn.Dropout(dropout)
        self.layers = ptn.ModuleList([layer.new() for _ in range(n_layers)])
        self.norm = ptn.LayerNorm(
            self.mod_dim,
            eps=layer.norm1.eps,
            elementwise_affine=layer.norm1.elementwise_affine,
            bias=layer.bias,
            device=device,
            dtype=dtype
        ) if layer.norm_first else Identity()
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

        out = self.drop(self.emb_pos_enc(self.embed(src)))
        for layer in self.layers:
            out = layer(out, attn_mask, is_causal)

        return self.finalize(self.norm(out)).transpose(-1, -2).contiguous(),

    def reset_parameters(self) -> None:
        for layer in self.layers:
            layer.reset_parameters()
        self.embed.reset_parameters()
        self.emb_pos_enc.reset_parameters()
        self.finalize.reset_parameters()
