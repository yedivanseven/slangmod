from swak.jsonobject import JsonObject
from swak.jsonobject.fields import Maybe, Lower
from ..enums import Scaling, Optimizers


class Train(JsonObject):
    batch_size: int = 64
    step_freq: int = 1
    clip_grad: float = 0.6
    label_smoothing: float = 0.1
    optimizer: Lower() = Optimizers.ADMAW
    max_epochs: int = 16
    patience: Maybe[int](int) = None
    learning_rate: float = 0.001
    warmup: int = 8_000
    scaling: Lower() = Scaling.COSINE
    power: float = 0.5
    gamma: float = 0.95
    cooldown: int = 100_000
    cb_freq: int = 1
