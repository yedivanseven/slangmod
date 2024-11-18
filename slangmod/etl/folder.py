import math
import torch.nn.functional as ptnf
from swak.misc import ArgRepr
from swak.pt.types import Tensor
from ..config import config

__all__ = [
    'SequenceFolder',
    'fold_train',
    'fold_test'
]


class SequenceFolder(ArgRepr):

    def __init__(self, seq_len: int, stride: int = 0) -> None:
        self.seq_len = seq_len
        self.stride = max(0, min(stride, seq_len))
        super().__init__(seq_len, self.stride)

    def n(self, length: int) -> int:
        if self.stride > 0:
            multiple = (length - self.seq_len - self.stride) / self.stride
        else:
            multiple = (length - (self.seq_len + 1)) / self.seq_len
        return 1 + math.ceil(multiple)

    @property
    def width(self) -> int:
        return self.seq_len + self.stride

    def padding(self, sequence: Tensor) -> int:
        length = sequence.size(0)
        if self.stride > 0:
            return self.seq_len + self.n(length) * self.stride - length
        else:
            return self.n(length) * self.seq_len + 1 - length

    def __call__(self, sequence: Tensor) -> Tensor:
        padded = ptnf.pad(sequence, (0, self.padding(sequence)), value=0)
        if self.stride > 0:
            return padded.unfold(0, self.width, self.stride)
        return padded.unfold(0, self.seq_len + 1, self.seq_len)


fold_train = SequenceFolder(config.data.seq_len, config.data.stride)
fold_test = SequenceFolder(config.data.seq_len, 0)
