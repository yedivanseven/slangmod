from swak.pt.misc import Compile
from .attention import Attention
from .layer import Layer
from .former import Former, vanilla_former
from .positions import Learnable, Sinusoidal
from ...config import config

__all__ = [
    'Attention',
    'Layer',
    'Former',
    'Learnable',
    'Sinusoidal',
    'vanilla_former',
    'compile_model'
]

compile_model = Compile(
    inplace=True,
    model=vanilla_former,
    disable=config.model.disable
)
