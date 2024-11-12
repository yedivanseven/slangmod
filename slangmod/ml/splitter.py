import torch as pt
from swak.misc import ArgRepr
from swak.pt.types import Tensor, Tensors3T
from ..config import config


class DataSplitter(ArgRepr):

    def __init__(
            self,
            seq_len: int,
            stride: int = 1,
            test: float = 0.1,
    ) -> None:
        self.seq_len = seq_len
        if stride < 1.0:
            self.stride = round(max(1.0, stride * seq_len))
        else:
            self.stride = round(min(stride, seq_len))
        self.test = test
        super().__init__(seq_len, self.stride, test)

    @property
    def train(self) -> float:
        return 1.0 - 2 * self.test

    def __call__(self, data: Tensor) -> Tensors3T:
        sequences = data.unfold(0, self.seq_len + 1, self.stride)
        n = sequences.shape[0]
        rand = pt.randperm(n, device=sequences.device, dtype=pt.long)
        test_index = int(n * self.train)
        validation_index = int(n * (self.train + self.test))
        return (
            sequences[rand][:test_index],
            sequences[rand][test_index:validation_index],
            sequences[rand][validation_index:]
        )


split_data = DataSplitter(
    seq_len=config.data.seq_len,
    stride=config.data.stride,
    test=config.data.test
)
