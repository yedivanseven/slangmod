import torch as pt
from swak.misc import ArgRepr
from swak.pt.types import Tensor, Tensors3T
from ..config import config


class DataSplitter(ArgRepr):

    def __init__(
            self,
            seq_len: int,
            test: float = 0.1,
            shuffle: bool = True
    ) -> None:
        self.seq_len = seq_len
        self.test = test
        self.shuffle = shuffle
        super().__init__(seq_len, test, shuffle)

    @property
    def stride(self) -> int:
        return self.seq_len + 1

    @property
    def train(self) -> float:
        return 1.0 - 2 * self.test

    def __call__(self, data: Tensor) -> Tensors3T:
        sequences = data.unfold(0, self.stride, self.stride)
        n = sequences.size(0)
        n_train = int(n * self.train)
        n_test = int(n * self.test)

        max_start = int(2 * n * self.test) + 1
        if self.shuffle:
            start = pt.randint(0, max_start, [1], device=sequences.device)
            indices = pt.randperm(max_start, device=sequences.device)
        else:
            start = max_start
            indices = pt.arange(max_start, device=sequences.device)
        stop = start + n_train

        train = sequences[start:stop]
        remainder = pt.cat([sequences[:start], sequences[stop:]], dim=0)

        test = remainder[indices][:n_test]
        validation = remainder[indices][n_test:]

        return train, test, validation


split_data = DataSplitter(
    seq_len=config.data.seq_len,
    test=config.data.test,
    shuffle=config.data.shuffle
)
