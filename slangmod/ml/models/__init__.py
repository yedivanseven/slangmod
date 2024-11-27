from swak.pt.misc import Compile
from .attention import Attention
from .layer import Layer
from .former import Former, vanilla_former
from .positions import Learnable, Sinusoidal, positions
from ...config import config

__all__ = [
    'Attention',
    'Layer',
    'Former',
    'Learnable',
    'Sinusoidal',
    'positions',
    'compile_model'
]

compile_model = Compile(
    inplace=True,
    model=vanilla_former,
    disable=config.model.disable
)
# ToDo: Play with a few strategically placed "contiguous" for speed!
