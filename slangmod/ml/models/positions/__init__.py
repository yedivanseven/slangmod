from swak.pt.misc import Identity
from .sinusoidal import Sinusoidal
from .learnable import Learnable
from .rotary import Rotary
from ....config import config, Positions

__all__ = [
    'Sinusoidal',
    'Learnable',
    'Rotary',
    'emb_pos_enc',
    'src_pos_enc',
    'qk_pos_enc'
]

emb_pos_enc = {
    Positions.SINUSOIDAL: Sinusoidal,
    Positions.LEARNABLE: Learnable,
    Positions.NONE: Identity
}[config.model.emb_pos_enc](
    mod_dim=config.model.dim,
    context=config.model.context,
    n_heads=config.model.n_heads,
    device=config.data.device,
    dtype=config.data.dtype,
)

src_pos_enc = {
    Positions.SINUSOIDAL: Sinusoidal,
    Positions.LEARNABLE: Learnable,
    Positions.NONE: Identity
}[config.model.src_pos_enc](
    mod_dim=config.model.dim,
    context=config.model.context,
    n_heads=config.model.n_heads,
    device=config.data.device,
    dtype=config.data.dtype,
)

qk_pos_enc = {
    Positions.ROTARY: Rotary,
    Positions.NONE: Identity
}[config.model.qk_pos_enc](
    mod_dim=config.model.dim,
    context=config.model.context,
    n_heads=config.model.n_heads,
    device=config.data.device,
    dtype=config.data.dtype,
)
