import torch as pt
from swak.jsonobject import JsonObject
from swak.jsonobject.fields import Lower
from swak.pt import device
from swak.pt.types import Dtype
from ..enums import Devices


class Data(JsonObject):
    device: Lower() = device.type
    seq_len: int = 512
    stride: int = 384
    jitter: int = 32
    test: float = 0.01
    shuffle: bool = True

    @property
    def dtype(self) -> Dtype:
        return pt.float32  #pt.bfloat16 if self.device == Devices.CUDA else pt.float32
