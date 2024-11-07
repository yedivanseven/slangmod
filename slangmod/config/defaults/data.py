import torch as pt
from swak.jsonobject import JsonObject
from swak.jsonobject.fields import Lower
from swak.pt import device
from swak.pt.types import Dtype
from ..enums import Devices


class Data(JsonObject):
    device: Lower() = device.type
    seq_len: int = 1024
    step: int = 1
    test: float = 0.005
    validate: float = 0.005

    @property
    def dtype(self) -> Dtype:
        return pt.bfloat16 if self.device == Devices.CUDA else pt.float32
