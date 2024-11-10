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
            device: Device | LiteralDevice,
            dtype: Dtype
    ) -> None:
        self.seqs = seqs
        self.device = pt.device(device)
        self.dtype = dtype
        self.mask = ptn.Transformer.generate_square_subsequent_mask(
            self.seq_len,
            device=self.device,
            dtype=dtype
        )
        self.__rand = pt.randperm(self.n, device=seqs.device, dtype=pt.long)

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        return f'{cls}(n={self.n})'

    @property
    def n(self) -> int:
        return self.seqs.shape[0]

    @property
    def seq_len(self) -> int:
        return self.seqs.shape[1] - 1

    def sample(self, batch_size: int, max_n: int | None = None) -> Batches:
        n = self.n if max_n is None else min(max_n, self.n)
        n_batches = math.ceil(n / batch_size)
        data = self.seqs[self.__rand[:n]]
        return iter(
            (
                (
                    data[batch * batch_size:(batch + 1) * batch_size, :-1].to(
                        self.device,
                        non_blocking=True
                    ),
                    self.mask,
                    None,
                    True
                ),
                data[batch * batch_size:(batch + 1) * batch_size, 1:].to(
                    self.device,
                    non_blocking=True
                )
            )
            for batch in range(n_batches)
        )


# ToDo: Increase sequence length during warmup, then randomize sequence length
class TrainData(TrainDataBase):

    def __init__(
            self,
            seqs: Tensor,
            device: Device | LiteralDevice,
            dtype: Dtype
    ) -> None:
        self.seqs = seqs
        self.device = pt.device(device)
        self.dtype = dtype
        self.mask = ptn.Transformer.generate_square_subsequent_mask(
            self.seq_len,
            device=self.device,
            dtype=dtype
        )
        self.__rand = pt.randperm(self.n, device=seqs.device, dtype=pt.long)

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        return f'{cls}(n={self.n})'

    @property
    def n(self) -> int:
        return self.seqs.shape[0]

    @property
    def seq_len(self) -> int:
        return self.seqs.shape[1] - 1

    def sample(self, batch_size: int, max_n: int | None = None) -> Batches:
        n = self.n if max_n is None else min(max_n, self.n)
        n_batches = math.ceil(n / batch_size)
        data = self.seqs[self.__rand[:n]]
        return iter(
            (
                (
                    data[batch * batch_size:(batch + 1) * batch_size, :-1].to(
                        self.device,
                        non_blocking=True
                    ),
                    self.mask,
                    None,
                    True
                ),
                data[batch * batch_size:(batch + 1) * batch_size, 1:].to(
                    self.device,
                    non_blocking=True
                )
            )
            for batch in range(n_batches)
        )

    def __call__(self, batch_size: int, step_freq: int = 1) -> Batches:
        n = self.n_for(batch_size, step_freq)
        rand = pt.randperm(self.n, device=self.seqs.device, dtype=pt.long)
        data = self.seqs[rand[:n]]
        return iter(
            (
                (
                    data[batch * batch_size:(batch + 1) * batch_size, :-1].to(
                        self.device,
                        non_blocking=True
                    ),
                    self.mask,
                    None,
                    True
                ),
                data[batch * batch_size:(batch + 1) * batch_size, 1:].to(
                    self.device,
                    non_blocking=True
                )
            )
            for batch in range(self.n_batches_of(batch_size, step_freq))
        )


make_train_data = Curry[TrainData](
    TrainData,
    config.data.device,
    config.data.dtype
)
make_test_data = Curry[TestData](
    TestData,
    config.data.device,
    config.data.dtype
)
make_validation_data = Curry[TestData](
    TestData,
    config.data.device,
    config.data.dtype
)
