import torch as pt
from swak.jsonobject import JsonObject
from swak.jsonobject.fields import Lower
from swak.pt import device
from swak.pt.types import Dtype
from ..enums import Devices


class Data(JsonObject):
    device: Lower() = device.type
    context: int = 1024  # 512  # 1024
    test: float = 0.01
    validate: float = 0.01

    @property
    def dtype(self) -> Dtype:
        return pt.float32 if self.device == Devices.CPU else pt.bfloat16
