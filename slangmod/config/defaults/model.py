from swak.jsonobject import JsonObject
from swak.jsonobject.fields import Lower
from ..enums import Positions, FeedForward


# ToDo: Resolve the positional encoding cases
# ToDo: Make choices nicer
class Model(JsonObject):
    dim: int = 64
    reference: bool = True
    emb_pos_enc: Lower() = Positions.SINUSOIDAL
    src_pos_enc: Lower() = Positions.NONE
    qk_pos_enc: Lower() = Positions.NONE
    context: int = 4096
    n_heads: int = 2
    n_layers: int = 2
    feedforward_flavour: Lower() = FeedForward.REFERENCE
    feedforward_factor: int = 4
    scale_grad_by_freq: bool = True
    dropout: float = 0.1
    bias: bool = True
    norm_first: bool = True
    compile: bool = True

    @property
    def disable(self) -> bool:
        return not self.compile
