from .sinusoidal import Sinusoidal
from .learnable import Learnable
from .rotary import Rotary
from ....config import config, Positions

__all__ = [
    'Sinusoidal',
    'Learnable',
    'positions'
]

positions = {
    Positions.SINUSOIDAL: Sinusoidal,
    Positions.ROTARY: Rotary,
    Positions.LEARNABLE: Learnable
}[config.model.positions](
    mod_dim=config.model.dim,
    context=config.model.context,
    n_heads=config.model.n_heads,
    device=config.data.device,
    dtype=config.data.dtype,
)
