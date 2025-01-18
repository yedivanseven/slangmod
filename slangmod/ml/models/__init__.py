from swak.pt.misc import Compile
from .attention import SelfAttention
from .layer import EncoderLayer, encoder_layer
from .encoder import Encoder
from .positions import Learnable, Sinusoidal, Rotary, emb_pos_enc
from .reference import Reference
from ..tokenizers import special
from ...config import config

__all__ = [
    'Reference',
    'SelfAttention',
    'EncoderLayer',
    'Encoder',
    'Learnable',
    'Sinusoidal',
    'Rotary',
    'compile_model'
]

# ToDo: Move this to the init of "ml"
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
    feedforward_factor=config.model.feedforward.factor,
    scale_grad_by_freq=config.model.scale_grad_by_freq,
    dropout=config.model.dropout,
    bias=config.model.bias,
    norm_first=config.model.norm_first,
    device=config.data.device,
    dtype=config.data.dtype
) if config.model.reference else Encoder(
    vocab=config.tokens.vocab,
    pad_id=special.pad_id,
    n_layers=config.model.n_layers,
    layer=encoder_layer,
    pos_enc=emb_pos_enc,
    bias=config.model.bias,
    dropout=config.model.dropout,
    scale_grad_by_freq=config.model.scale_grad_by_freq,
    device=config.data.device,
    dtype=config.data.dtype
)

# ToDo: Maybe compile right away to not keep two copies?
compile_model = Compile(
    inplace=True,
    model=model,
    disable=config.model.disable
)
