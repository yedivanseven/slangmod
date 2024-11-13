import math
import torch as pt
import torch.nn as ptn
from swak.pt.types import Tensor, Dtype, Device
from swak.pt.train import TestDataBase, TrainDataBase
from swak.funcflow import Curry
from ..config import config, LiteralDevice
from .types import Batches


class TestData(TestDataBase):

    def __init__(
            self,
            seqs: Tensor,
            shuffle: bool,
            device: Device | LiteralDevice,
            dtype: Dtype
    ) -> None:
        self.seqs = seqs
        self.shuffle = shuffle
        self.device = pt.device(device)
        self.dtype = dtype
        self.mask = ptn.Transformer.generate_square_subsequent_mask(
            self.seq_len,
            device=self.device,
            dtype=dtype
        )
        if self.shuffle:
            self.__indices = pt.randperm(self.n, device=seqs.device)
        else:
            self.__indices = pt.arange(self.n, device=seqs.device)

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        return f'{cls}(n={self.n})'

    @property
    def n(self) -> int:
        return self.seqs.size(0)

    @property
    def seq_len(self) -> int:
        return self.seqs.size(1) - 1

    def sample(self, batch_size: int, max_n: int | None = None) -> Batches:
        n = self.n if max_n is None else min(max_n, self.n)
        batches = range(math.ceil(n / batch_size))
        seqs = self.seqs[self.__indices[:n]]
        return iter(
            (
                (
                    seqs[batch * batch_size:(batch + 1) * batch_size, :-1].to(
                        self.device,
                        non_blocking=True
                    ),
                    self.mask,
                    None,
                    True
                ),
                seqs[batch * batch_size:(batch + 1) * batch_size, 1:].to(
                    self.device,
                    non_blocking=True
                )
            )
            for batch in batches
        )


class TrainData(TrainDataBase):

    def __init__(
            self,
            seqs: Tensor,
            stride: int,
            shuffle: bool,
            device: Device | LiteralDevice,
            dtype: Dtype
    ) -> None:
        self.__n = self.__full_n = seqs.size(0)
        self.__seq_len = seqs.size(1) - 1
        self.stride = stride
        self.data = self.fold(seqs)
        self.seqs = self.data.unfold(0, self.seq_len + 1, self.stride)
        self.shuffle = shuffle
        self.device = pt.device(device)
        self.dtype = dtype
        self.mask = ptn.Transformer.generate_square_subsequent_mask(
            self.seq_len,
            device=self.device,
            dtype=dtype
        )
        if self.shuffle:
            self.__indices = pt.randperm(self.__full_n, device=seqs.device)
        else:
            self.__indices = pt.arange(self.__full_n, device=seqs.device)

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        return f'{cls}(n={self.n})'

    @property
    def n(self) -> int:
        return self.seqs.size(0)

    @property
    def seq_len(self) -> int:
        return self.seqs.size(1) - 1

    @property
    def max_start(self) -> int:
        return self.seq_len - self.stride

    def fold(self, seqs: Tensor) -> Tensor:
        n_tokens = (self.__full_n - 1) * self.stride + self.seq_len + 1

        seq = pt.zeros(n_tokens, dtype=seqs.dtype, device=seqs.device)
        count = pt.zeros(n_tokens, dtype=seqs.dtype, device=seqs.device)

        for i in range(self.__full_n):
            start = i * self.stride
            stop = start + self.seq_len + 1
            seq[start:stop] += seqs[i]
            count[start:stop] += 1

        seq //= count
        return seq

    def sample(self, batch_size: int, max_n: int | None = None) -> Batches:
        n = self.__full_n if max_n is None else min(max_n, self.__full_n)
        batches = range(math.ceil(n / batch_size))
        seqs = self.seqs[self.__indices[:n]]
        return iter(
            (
                (
                    seqs[batch * batch_size:(batch + 1) * batch_size, :-1].to(
                        self.device,
                        non_blocking=True
                    ),
                    self.mask,
                    None,
                    True
                ),
                seqs[batch * batch_size:(batch + 1) * batch_size, 1:].to(
                    self.device,
                    non_blocking=True
                )
            )
            for batch in batches
        )

    def __call__(
            self,
            batch_size: int,
            step_freq: int = 1,
            epoch: int = 0
    ) -> Batches:
        if self.shuffle:
            start = pt.randint(0, self.max_start, [1], device=self.data.device)
            seqs = self.data[start:].unfold(0, self.seq_len + 1, self.stride)
            self.__n = seqs.size(0)
            n = self.n_for(batch_size, step_freq)
            random = pt.randperm(self.n, device=self.data.device)
            seqs = seqs[random[:n]]
        else:
            seqs = self.seqs
        batches = range(self.n_batches_of(batch_size, step_freq))
        return iter(
            (
                # Source sequence, attention mask, and is_causal flag
                (
                    seqs[batch * batch_size:(batch + 1) * batch_size, :-1].to(
                        self.device,
                        non_blocking=True
                    ),
                    self.mask,
                    None,  # The padding/unknown mask is None for training
                    True
                ),
                # Target sequence, shifted by one relative to the source
                seqs[batch * batch_size:(batch + 1) * batch_size, 1:].to(
                    self.device,
                    non_blocking=True
                )
            )
            for batch in batches
        )


make_train_data = Curry[TrainData](
    TrainData,
    stride=config.data.stride,
    shuffle=config.data.shuffle,
    device=config.data.device,
    dtype=config.data.dtype
)
make_test_data = Curry[TestData](
    TestData,
    shuffle=config.data.shuffle,
    device=config.data.device,
    dtype=config.data.dtype
)
make_validation_data = Curry[TestData](
    TestData,
    shuffle=config.data.shuffle,
    device=config.data.device,
    dtype=config.data.dtype
)
