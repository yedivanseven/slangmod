from swak.jsonobject import JsonObject
from swak.jsonobject.fields import Maybe


class Train(JsonObject):
    batch_size: int = 128
    step_freq: int = 1
    label_smoothing: float = 0.05
    learning_rate: Maybe[float](float) = 0.004
    power: float = 0.5
    gamma: float = 0.95
    max_epochs: int = 1024
    warmup: int = 2000
    patience: Maybe[int](int) = 1

    @property
    def super_batch(self) -> int:
        return self.batch_size * self.step_freq
