from .sinusoidal import Sinusoidal
from .learnable import Learnable
from ....config import config, Positions

__all__ = [
    'Sinusoidal',
    'Learnable',
    'positions'
]

positions = {
    Positions.SINUSOIDAL: Sinusoidal,
    Positions.LEARNABLE: Learnable
}[config.model.positions](
    mod_dim=config.model.dim,
    context=config.model.context,
    device=config.data.device,
    dtype=config.data.dtype,
)
