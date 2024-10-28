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
            self.context,
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
    def context(self) -> int:
        return self.seqs.shape[1] - 1

    @property
    def pin(self) -> bool:
        return self.device == 'cuda'

    def sample(self, batch_size: int, max_n: int | None = None) -> Batches:
        n = self.n if max_n is None else min(max_n, self.n)
        srcs = self.seqs[self.__rand[:n], :-1].contiguous().split(batch_size)
        tgts = self.seqs[self.__rand[:n], 1:].contiguous().split(batch_size)
        if self.pin:
            srcs = tuple(batch.pin_memory(self.device) for batch in srcs)
            tgts = tuple(batch.pin_memory(self.device) for batch in tgts)
        return iter(
            (
                (
                    src.to(self.device, non_blocking=True),
                    self.mask,
                    None,
                    True
                ),
                tgt.to(self.device, non_blocking=True)
            )
            for src, tgt in zip(srcs, tgts)
        )


class TrainData(TrainDataBase):

    def __init__(self, seqs: Tensor, device: Device, dtype: Dtype) -> None:
        self.seqs = seqs
        self.device = device
        self.dtype = dtype
        self.mask = ptn.Transformer.generate_square_subsequent_mask(
            self.context,
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
    def context(self) -> int:
        return self.seqs.shape[1] - 1

    @property
    def pin(self) -> bool:
        return self.device == 'cuda'

    def sample(self, batch_size: int, max_n: int | None = None) -> Batches:
        n = self.n if max_n is None else min(max_n, self.n)
        srcs = self.seqs[self.__rand[:n], :-1].contiguous().split(batch_size)
        tgts = self.seqs[self.__rand[:n], 1:].contiguous().split(batch_size)
        if self.pin:
            srcs = tuple(batch.pin_memory(self.device) for batch in srcs)
            tgts = tuple(batch.pin_memory(self.device) for batch in tgts)
        return iter(
            (
                (
                    src.to(self.device, non_blocking=True),
                    self.mask,
                    None,
                    True
                ),
                tgt.to(self.device, non_blocking=True)
            )
            for src, tgt in zip(srcs, tgts)
        )

    def __call__(self, batch_size: int) -> Batches:
        rand = pt.randperm(self.n, device=self.seqs.device, dtype=pt.long)
        srcs = self.seqs[rand, :-1].contiguous().split(batch_size)
        tgts = self.seqs[rand, 1:].contiguous().split(batch_size)
        if self.pin:
            srcs = tuple(batch.pin_memory(self.device) for batch in srcs)
            tgts = tuple(batch.pin_memory(self.device) for batch in tgts)
        return iter(
            (
                (
                    src.to(self.device, non_blocking=True),
                    self.mask,
                    None,
                    True
                ),
                tgt.to(self.device, non_blocking=True)
            )
            for src, tgt in zip(srcs, tgts)
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
