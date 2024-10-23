import torch as pt
import torch.nn as ptn
from swak.pt.types import Tensor, Dtype
from swak.pt.train import TestDataBase, TrainDataBase
from swak.funcflow import Curry
from ..config import config
from .types import Batches


class TestData(TestDataBase):

    def __init__(self, sequences: Tensor, dtype: Dtype) -> None:
        self.sequences = sequences
        self.dtype = dtype
        self.mask = ptn.Transformer.generate_square_subsequent_mask(
            self.context,
            device=sequences.device,
            dtype=dtype
        )
        self.__rand = pt.randperm(self.n)

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        return f'{cls}(n={self.n})'

    @property
    def n(self) -> int:
        return self.sequences.shape[0]

    @property
    def context(self) -> int:
        return self.sequences.shape[1] - 1

    def sample(self, batch_size: int, max_n: int | None = None) -> Batches:
        n = self.n if max_n is None else min(max_n, self.n)
        srcs = self.sequences[self.__rand[:n], :-1].split(batch_size)
        tgts = self.sequences[self.__rand[:n], 1:].split(batch_size)
        zipped = zip(srcs, tgts)
        return iter(((src, self.mask, None, True), tgt) for src, tgt in zipped)


class TrainData(TrainDataBase):

    def __init__(self, sequences: Tensor, dtype: Dtype) -> None:
        self.sequences = sequences
        self.dtype = dtype
        self.mask = ptn.Transformer.generate_square_subsequent_mask(
            self.context,
            device=sequences.device,
            dtype=dtype
        )
        self.__rand = pt.randperm(self.n)

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        return f'{cls}(n={self.n})'

    @property
    def n(self) -> int:
        return self.sequences.shape[0]

    @property
    def context(self) -> int:
        return self.sequences.shape[1] - 1

    def sample(self, batch_size: int, max_n: int | None = None) -> Batches:
        n = self.n if max_n is None else min(max_n, self.n)
        srcs = self.sequences[self.__rand[:n], :-1].split(batch_size)
        tgts = self.sequences[self.__rand[:n], 1:].split(batch_size)
        zipped = zip(srcs, tgts)
        return iter(((src, self.mask, None, True), tgt) for src, tgt in zipped)

    def __call__(self, batch_size: int) -> Batches:
        rand = pt.randperm(self.n)
        srcs = self.sequences[rand, :-1].split(batch_size)
        tgts = self.sequences[rand, 1:].split(batch_size)
        zipped = zip(srcs, tgts)
        return iter(((src, self.mask, None, True), tgt) for src, tgt in zipped)


make_train_data = Curry[TrainData](
    TrainData,
    config.data.dtype
)
make_test_data = Curry[TestData](
    TestData,
    config.data.dtype
)
make_validation_data = Curry[TestData](
    TestData,
    config.data.dtype
)
