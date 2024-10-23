from swak.jsonobject import JsonObject
from swak.jsonobject.fields import Maybe


class Train(JsonObject):
    batch_size: int = 256  # 128
    label_smoothing: float = 0.1
    learning_rate: float = 0.01
    max_epochs: int = 1024
    warmup: int = 5  # 10
    patience: Maybe[int](int) = 5  # 10
    max_n: Maybe[int](int) = None
