import torch as pt
import torch.nn as ptn
from swak.pt.types import Tensor, Dtype
from swak.pt.train import TestDataBase, TrainDataBase
from swak.funcflow import Curry
from ..config import config
from .types import Batch, Batches


class TestData(TestDataBase):

    def __init__(self, data: Tensor, context: int, dtype: Dtype) -> None:
        self.data = data.unfold(0, context + 1, 1)
        self.context = context
        self.dtype = dtype
        self.mask = ptn.Transformer.generate_square_subsequent_mask(
            context,
            device=data.device,
            dtype=dtype
        )
        self.__indices = pt.randperm(self.n - self.context)

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        return f'{cls}(n={self.n})'

    @property
    def n(self) -> int:
        return self.data.shape[0]

    def sample(self, size: int) -> Batch:
        return (
            (self.data[self.__indices[:size], :-1], self.mask, None, True),
             self.data[self.__indices[:size], 1:]
        )


class TrainData(TrainDataBase):

    def __init__(self, data: Tensor, context: int, dtype: Dtype) -> None:
        self.data = data.unfold(0, context + 1, 1)
        self.context = context
        self.dtype = dtype
        self.mask = ptn.Transformer.generate_square_subsequent_mask(
            context,
            device=data.device,
            dtype=dtype
        )
        self.__indices = pt.randperm(self.n - self.context)

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        return f'{cls}(n={self.n})'

    @property
    def n(self) -> int:
        return self.data.shape[0]

    def sample(self, size: int) -> Batch:
        return (
            (self.data[self.__indices[:size], :-1], self.mask, None, True),
             self.data[self.__indices[:size], 1:]
        )

    def __call__(self, batch_size: int) -> Batches:
        indices = pt.randperm(self.n - self.context)
        srcs = self.data[indices, :-1].split(batch_size)
        tgts = self.data[indices, 1:].split(batch_size)
        zipped = zip(srcs, tgts)
        return iter(((src, self.mask, None, True), tgt) for src, tgt in zipped)


make_train_data = Curry[TrainData](
    TrainData,
    config.context,
    config.dtype
)
make_test_data = Curry[TestData](
    TestData,
    config.context,
    config.dtype
)
make_validation_data = Curry[TestData](
    TestData,
    config.context,
    config.dtype
)
