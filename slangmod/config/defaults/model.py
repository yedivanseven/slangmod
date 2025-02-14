from swak.jsonobject import JsonObject
from swak.jsonobject.fields import Lower
from ..enums import Positions, FeedForwards, Activations, Gates, Norms


class FeedForward(JsonObject):
    flavor: Lower() = FeedForwards.VANILLA
    activation: Lower() = Activations.GELU
    gate: Lower() = Gates.GELU
    bias: bool = False
    factor: int = 4


class Model(JsonObject):
    dim: int = 512
    scale_grad_by_freq: bool = True
    positions: Lower() = Positions.VANILLA
    context: int = 4096
    n_heads: int = 8
    n_layers: int = 8
    attn_bias: bool = False
    feedforward: FeedForward = FeedForward()
    dropout: float = 0.1
    norm_cls: str = Norms.LAYER
    norm_bias: bool = True
    norm_first: bool = True
    compile: bool = True

    @property
    def disable(self) -> bool:
        """Negation of `compile` for consistency in notation."""
        return not self.compile

# ToDo. Rethink defaults for model size. Change documentation accordingly
