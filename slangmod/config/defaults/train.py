from swak.jsonobject import JsonObject
from swak.jsonobject.fields import Maybe


class Train(JsonObject):
    batch_size: int = 256  # 256  # 128
    label_smoothing: float = 0.0
    learning_rate: float = 0.005
    max_epochs: int = 1024
    warmup: int = 10  # 5  # 10
    patience: Maybe[int](int) = 10  # 5  # 10
    max_n: Maybe[int](int) = None
