import math
import torch as pt
import torch.nn as ptn
from swak.pt.types import Tensor, Dtype
from swak.pt.train import TestDataBase, TrainDataBase
from swak.funcflow import Curry
from ..config import config
from .types import Batches, Device


class TestData(TestDataBase):

    def __init__(self, seqs: Tensor, device: Device, dtype: Dtype) -> None:
        self.seqs = seqs
        self.device = device
        self.dtype = dtype
        self.mask = ptn.Transformer.generate_square_subsequent_mask(
            self.seq_len,
            device=pt.device(device),
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

    @property
    def pin(self) -> bool:
        return self.device == 'cuda'

    def sample(self, batch_size: int, max_n: int | None = None) -> Batches:
        n = self.n if max_n is None else min(max_n, self.n)
        n_batches = math.ceil(n / batch_size)
        srcs = self.seqs[self.__rand[:n], :-1].contiguous()
        tgts = self.seqs[self.__rand[:n], 1:].contiguous()
        if self.pin:
            srcs = srcs.pin_memory(self.device)
            tgts = tgts.pin_memory(self.device)
        return iter(
            (
                (
                    srcs[batch * batch_size:(batch + 1) * batch_size].to(
                        self.device,
                        non_blocking=True
                    ),
                    self.mask,
                    None,
                    True
                ),
                tgts[batch * batch_size:(batch + 1) * batch_size].to(
                    self.device,
                    non_blocking=True
                )
            )
            for batch in range(n_batches)
        )


class TrainData(TrainDataBase):

    def __init__(
            self,
            seqs: Tensor,
            step_freq: int,
            device: Device,
            dtype: Dtype
    ) -> None:
        self.seqs = seqs
        self.step_freq = step_freq
        self.device = device
        self.dtype = dtype
        self.mask = ptn.Transformer.generate_square_subsequent_mask(
            self.seq_len,
            device=pt.device(device),
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

    @property
    def pin(self) -> bool:
        return self.device == 'cuda'

    def sample(self, batch_size: int, max_n: int | None = None) -> Batches:
        n = self.n if max_n is None else min(max_n, self.n)
        n_batches = math.ceil(n / batch_size)
        srcs = self.seqs[self.__rand[:n], :-1].contiguous()
        tgts = self.seqs[self.__rand[:n], 1:].contiguous()
        if self.pin:
            srcs = srcs.pin_memory(self.device)
            tgts = tgts.pin_memory(self.device)
        return iter(
            (
                (
                    srcs[batch * batch_size:(batch + 1) * batch_size].to(
                        self.device,
                        non_blocking=True
                    ),
                    self.mask,
                    None,
                    True
                ),
                tgts[batch * batch_size:(batch + 1) * batch_size].to(
                    self.device,
                    non_blocking=True
                )
            )
            for batch in range(n_batches)
        )

    def _max_n_for(self, batch_size: int) -> int:
        if self.step_freq <= 1:
            return self.n
        super_batch_size = self.step_freq * batch_size
        return super_batch_size * (self.n // super_batch_size)

    def __call__(self, batch_size: int) -> Batches:
        max_n = self._max_n_for(batch_size)
        n_batches = math.ceil(max_n / batch_size)
        rand = pt.randperm(self.n, device=self.seqs.device, dtype=pt.long)
        srcs = self.seqs[rand[:max_n], :-1].contiguous()
        tgts = self.seqs[rand[:max_n], 1:].contiguous()
        if self.pin:
            srcs = srcs.pin_memory(self.device)
            tgts = tgts.pin_memory(self.device)
        return iter(
            (
                (
                    srcs[batch * batch_size:(batch + 1) * batch_size].to(
                        self.device,
                        non_blocking=True
                    ),
                    self.mask,
                    None,
                    True
                ),
                tgts[batch * batch_size:(batch + 1) * batch_size].to(
                    self.device,
                    non_blocking=True
                )
            )
            for batch in range(n_batches)
        )


make_train_data = Curry[TrainData](
    TrainData,
    config.train.step_freq,
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
