import torch as pt
from swak.jsonobject import JsonObject
from swak.jsonobject.fields import Lower
from swak.pt import device
from swak.pt.types import Dtype
from ..enums import Devices


class Data(JsonObject):
    device: Lower() = device.type
    seq_len: int = 512
    overlap: float = 0.25
    jitter: int = 32
    shuffle: bool = True

    @property
    def dtype(self) -> Dtype:
        """Heuristic dtype to exploit device capabilities."""
        return pt.bfloat16 if self.device == Devices.CUDA else pt.float32
