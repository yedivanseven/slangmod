from .sinusoidal import Sinusoidal, sinusoidal
from .learnable import Learnable, learnable
from ....config import config, Positions

__all__ = [
    'Sinusoidal',
    'Learnable',
    'positions'
]

positions = {
    Positions.SINUSOIDAL: sinusoidal,
    Positions.LEARNABLE: learnable
}[config.model.positions]
