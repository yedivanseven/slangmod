from swak.jsonobject import JsonObject
from swak.jsonobject.fields import Maybe


class Train(JsonObject):
    batch_size: int = 32
    step_freq: int = 1
    label_smoothing: float = 0.1
    learning_rate: Maybe[float](float) = 0.01
    power: float = 0.5
    max_epochs: int = 1024
    warmup: int = 4000
    patience: Maybe[int](int) = 5
