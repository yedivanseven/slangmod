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
    Positions.VANILLA: Sinusoidal,
    Positions.LEARNABLE: Learnable,
    Positions.SINUSOIDAL: Identity,
    Positions.ROTARY: Identity
}[config.model.positions](
    mod_dim=config.model.dim,
    context=config.model.context,
    n_heads=config.model.n_heads,
    device=config.data.device,
    dtype=config.data.dtype,
)

src_pos_enc = {
    Positions.VANILLA: Identity,
    Positions.LEARNABLE: Identity,
    Positions.SINUSOIDAL: Sinusoidal,
    Positions.ROTARY: Identity
}[config.model.positions](
    mod_dim=config.model.dim,
    context=config.model.context,
    n_heads=config.model.n_heads,
    device=config.data.device,
    dtype=config.data.dtype,
)

qk_pos_enc = {
    Positions.VANILLA: Identity,
    Positions.LEARNABLE: Identity,
    Positions.SINUSOIDAL: Identity,
    Positions.ROTARY: Rotary
}[config.model.positions](
    mod_dim=config.model.dim,
    context=config.model.context,
    n_heads=config.model.n_heads,
    device=config.data.device,
    dtype=config.data.dtype,
)
