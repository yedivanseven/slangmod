import torch as pt
import torch.nn as ptn
from swak.pt.types import Tensor
from swak.pt.train import TestDataBase, TrainDataBase
from swak.funcflow import Partial
from ..config import config
from .types import Batch, Batches


class TestData(TestDataBase):

    def __init__(self, context: int, data: Tensor) -> None:
        self.context = context
        self.data = data.unfold(0, context + 1, 1)
        self.mask = ptn.Transformer.generate_square_subsequent_mask(
            context
        ).to(data.device)
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

    def __init__(self, context: int, data: Tensor) -> None:
        self.context = context
        self.data = data.unfold(0, context + 1, 1)
        self.mask = ptn.Transformer.generate_square_subsequent_mask(
            context
        ).to(data.device)
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


make_train_data = Partial[TrainData](TrainData, config.context)
make_test_data = Partial[TestData](TestData, config.context)
make_validation_data = Partial[TestData](TestData, config.context)
