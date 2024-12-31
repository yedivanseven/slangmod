from swak.pt.misc import Compile
from .attention import Attention
from .layer import Layer, layer
from .former import Former
from .positions import Learnable, Sinusoidal, Rotary, emb_pos_enc
from .reference import Reference
from ...config import config

__all__ = [
    'Reference',
    'Attention',
    'Layer',
    'Former',
    'Learnable',
    'Sinusoidal',
    'Rotary',
    'compile_model'
]

model = Reference(
    mod_dim=config.model.dim,
    vocab=config.tokens.vocab,
    n_heads=config.model.n_heads,
    n_layers=config.model.n_layers,
    pos_enc=Sinusoidal(
        mod_dim=config.model.dim,
        context=config.model.context,
        device=config.data.device,
        dtype=config.data.dtype
    ),
    feedforward_factor=config.model.feedforward_factor,
    scale_grad_by_freq=config.model.scale_grad_by_freq,
    dropout=config.model.dropout,
    bias=config.model.bias,
    norm_first=config.model.norm_first,
    device=config.data.device,
    dtype=config.data.dtype
) if config.model.reference else Former(
    mod_dim=config.model.dim,
    vocab=config.tokens.vocab,
    layer=layer,
    n_layers=config.model.n_layers,
    emb_pos_enc=emb_pos_enc,
    bias=config.model.bias,
    dropout=config.model.dropout,
    scale_grad_by_freq=config.model.scale_grad_by_freq,
    device=config.data.device,
    dtype=config.data.dtype
)

compile_model = Compile(
    inplace=True,
    model=model,
    disable=config.model.disable
)
