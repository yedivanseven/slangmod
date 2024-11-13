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
                # Source sequence, attention mask, and is_causal flag
                (
                    seqs[batch * batch_size:(batch + 1) * batch_size, :-1].to(
                        self.device,
                        non_blocking=True
                    ),
                    self.mask,
                    None,  # The padding/unknown mask is None for testing
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


class TrainData(TrainDataBase):

    def __init__(
            self,
            seqs: Tensor,
            stride: int,
            shuffle: bool,
            device: Device | LiteralDevice,
            dtype: Dtype
    ) -> None:
        self.seqs = seqs
        if stride < 1.0:
            self.stride = round(max(1.0, stride * self.seq_len))
        else:
            self.stride = round(min(stride, self.seq_len))
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

    @property
    def max_start(self) -> int:
        return self.seq_len - self.stride

    @property
    def data(self) -> Tensor:
        return self.seqs.ravel()

    def sample(self, batch_size: int, max_n: int | None = None) -> Batches:
        n = self.n if max_n is None else min(max_n, self.n)
        batches = range(math.ceil(n / batch_size))
        seqs = self.seqs[self.__indices[:n]]
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

    def __call__(
            self,
            batch_size: int,
            step_freq: int = 1,
            epoch: int = 0
    ) -> tuple[int, Batches]:
        if self.shuffle:
            start = pt.randint(0, self.max_start, [1], device=self.seqs.device)
            seqs = self.data[start:].unfold(0, self.seq_len + 1, self.stride)
            n = seqs.size(0)
            adjusted_n = self.adjust_n_for(batch_size, step_freq, n)
            shuffled = pt.randperm(n, device=self.data.device)
            seqs = seqs[shuffled[:adjusted_n]]
        else:
            seqs = self.data.unfold(0, self.seq_len + 1, self.stride)
            n = seqs.size(0)
            adjusted_n = self.adjust_n_for(batch_size, step_freq, n)
            seqs = seqs[:adjusted_n]
        n_batches = self.adjust_batches_for(batch_size, step_freq, n)
        return n_batches, iter(
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
            for batch in range(n_batches)
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
