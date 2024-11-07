import torch as pt
from swak.jsonobject import JsonObject
from swak.pt import device
from swak.pt.types import Dtype
from ..enums import Devices


class Data(JsonObject):
    device: pt.device = device
    seq_len: int = 1024
    step: int = 1
    test: float = 0.01
    validate: float = 0.01

    @property
    def dtype(self) -> Dtype:
        return pt.float32 if self.device.type == Devices.CPU else pt.bfloat16
