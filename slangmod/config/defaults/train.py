from swak.jsonobject import JsonObject
from swak.jsonobject.fields import Maybe, Lower
from ..enums import Scaling


class Train(JsonObject):
    batch_size: int = 32
    step_freq: int = 1
    label_smoothing: float = 0.1
    learning_rate: Maybe[float](float) = 0.01
    max_epochs: int = 1024
    warmup: int = 4000
    scaling: Lower() = Scaling.COSINE
    power: float = 0.5
    cooldown: int = 64
    patience: Maybe[int](int) = 10
