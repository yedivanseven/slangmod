import torch as pt
from swak.jsonobject import JsonObject
from swak.pt import device
from swak.pt.types import Dtype


class Data(JsonObject):
    context: int = 512  # 1024
    frac_test: float = 0.05
    frac_validate: float = 0.05

    @property
    def dtype(self) -> Dtype:
        return pt.float32 if device.type == 'cpu' else pt.bfloat16
