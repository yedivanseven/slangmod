from swak.jsonobject import JsonObject
from swak.jsonobject.fields import Lower


class Model(JsonObject):
    mod_dim: int = 64  # 32  # 64
    positions: Lower() = 'sinusoidal'
    n_heads: int = 2
    n_layers: int = 2  # 2  # 4
    scale_grad_by_freq: bool = True
    dropout: float = 0.1
    bias: bool = True
