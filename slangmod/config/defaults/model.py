from swak.jsonobject import JsonObject
from swak.jsonobject.fields import Lower
from ..enums import Positions, FeedForward


class FF(JsonObject):
    flavour: Lower() = FeedForward.VANILLA
    factor: int = 4


class Model(JsonObject):
    dim: int = 64
    reference: bool = True
    positions: Lower() = Positions.VANILLA
    context: int = 4096
    n_heads: int = 2
    n_layers: int = 2
    feedforward: FF = FF()
    scale_grad_by_freq: bool = True
    dropout: float = 0.1
    bias: bool = True
    norm_first: bool = True
    compile: bool = True

    @property
    def disable(self) -> bool:
        return not self.compile
