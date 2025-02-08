import torch as pt
from swak.jsonobject import JsonObject
from swak.jsonobject.fields import Lower
from swak.pt import device
from swak.pt.types import Dtype
from ..enums import Dtypes


class Data(JsonObject):
    device: Lower() = device.type
    precision: Lower() = 'bfloat16'
    seq_len: int = 512
    overlap: float = 0.25
    jitter: int = 32
    shuffle: bool = True

    @property
    def dtype(self) -> Dtype:
        """The given numerical precision translated to a PyTorch dtype."""
        return {
            Dtypes.FLOAT: pt.float,
            Dtypes.FLOAT32: pt.float32,
            Dtypes.BFLOAT: pt.bfloat16,
            Dtypes.BFLOAT16: pt.bfloat16
        }[self.precision]
