from swak.jsonobject import JsonObject
from swak.jsonobject.fields import Maybe


class Train(JsonObject):
    batch_size: int = 256  # 256  # 128
    grad_freq: int = 1
    label_smoothing: float = 0.0
    learning_rate: Maybe[float](float) = None
    power: float = 0.5
    gamma: float = 0.95
    max_epochs: int = 1024
    warmup: int = 10  # 5  # 10
    patience: Maybe[int](int) = 5  # 5  # 10
    max_n: Maybe[int](int) = None

    @property
    def drop_last(self) -> bool:
        return self.grad_freq > 1

    @property
    def lr(self) -> float:
        if self.learning_rate is None:
            return (0.01 / 512.0) * self.batch_size
        return self.learning_rate
