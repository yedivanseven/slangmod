from swak.jsonobject import JsonObject
from swak.jsonobject.fields import Maybe, Lower
from ..enums import Scaling, Optimizers


class Train(JsonObject):
    batch_size: int = 32
    step_freq: int = 1
    clip_grad: float = 0.8
    label_smoothing: float = 0.1
    learning_rate: Maybe[float](float) = 0.01
    max_epochs: int = 1024
    warmup: int = 4000
    batch_step: bool = True
    optimizer: Lower() = Optimizers.ADMAW
    scaling: Lower() = Scaling.COSINE
    power: float = 0.5
    gamma: float = 0.95
    cooldown: int = 30_000
    patience: Maybe[int](int) = 2
