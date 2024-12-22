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

    def __init__(self, seq_len: int, stride: int = 0, jitter: int = 0) -> None:
        self.seq_len = seq_len
        self.stride = max(0, min(stride, seq_len))
        self.jitter = min(self.stride, jitter)
        super().__init__(seq_len, self.stride, self.jitter)

    @property
    def width(self) -> int:
        if self.stride > 0:
            return self.seq_len + self.jitter
        return self.seq_len + 1

    def clamp(self, length: int) -> int:
        if self.stride > 0:
            return max(self.jitter + 1, length)
        return max(0, length - 2)

    def extra_rows_for(self, length: int) -> int:
        if self.stride > 0:
            return (self.clamp(length) - self.jitter - 1) // self.stride
        return self.clamp(length) // self.seq_len

    def padded_to(self, length: int) -> int:
        if self.stride > 0:
            return self.width + self.extra_rows_for(length) * self.stride
        return self.width + self.extra_rows_for(length) * self.seq_len

    def missing_for(self, length: int) -> int:
        return self.padded_to(length) - length

    def pad(self, sequence: Tensor) -> Tensor:
        n_pad = self.missing_for(sequence.size(0))
        return ptnf.pad(sequence, (0, n_pad), value=0)

    def __call__(self, sequence: Tensor) -> Tensor:
        if self.stride > 0:
            return self.pad(sequence).unfold(0, self.width, self.stride)
        return self.pad(sequence).unfold(0, self.width, self.seq_len)


fold_train = SequenceFolder(
    config.data.seq_len,
    config.data.stride,
    config.data.jitter
)
fold_test = SequenceFolder(config.data.seq_len)
