from swak.jsonobject import JsonObject
from swak.jsonobject.fields import Lower
from ..enums import Positions, FeedForwards, Activations, Gates


class FeedForward(JsonObject):
    flavour: Lower() = FeedForwards.VANILLA
    activation: Lower() = Activations.GELU
    gate: Lower() = Gates.GELU
    factor: int = 4


class Model(JsonObject):
    dim: int = 64
    reference: bool = False
    scale_grad_by_freq: bool = True
    positions: Lower() = Positions.VANILLA
    context: int = 4096
    n_heads: int = 2
    n_layers: int = 2
    feedforward: FeedForward = FeedForward()
    dropout: float = 0.1
    bias: bool = True
    norm_first: bool = True
    compile: bool = True

    @property
    def disable_compile(self) -> bool:
        return not self.compile
