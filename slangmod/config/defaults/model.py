from swak.jsonobject import JsonObject
from swak.jsonobject.fields import Lower
from ..enums import Positions


class Model(JsonObject):
    dim: int = 64
    positions: Lower() = Positions.SINUSOIDAL
    context: int = 8192
    n_heads: int = 2
    n_layers: int = 2
    feedforward_factor: int = 4
    scale_grad_by_freq: bool = True
    dropout: float = 0.1
    bias: bool = True
    norm_first: bool = False
    compile: bool = True

    @property
    def disable(self) -> bool:
        return not self.compile
