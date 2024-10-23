import torch as pt
from swak.misc import ArgRepr
from swak.pt.types import Tensor, Tensors3T
from ..config import config


class TrainTestValidationSplitter(ArgRepr):

    def __init__(self, context: int, test: float, validation: float) -> None:
        super().__init__(context, test, validation)
        self.context = context
        self.test = test
        self.validation = validation
        self.__rand = None

    @property
    def train(self) -> float:
        return 1.0 - self.test - self.validation

    def __call__(self, data: Tensor) -> Tensors3T:
        sequences = data.unfold(0, self.context + 1, 1)
        n = sequences.shape[0]
        self.__rand = pt.randperm(n) if self.__rand is None else self.__rand
        test_index = int(n * self.train)
        validation_index = int(n * (self.train + self.test))
        return (
            sequences[self.__rand][:test_index],
            sequences[self.__rand][test_index:validation_index],
            sequences[self.__rand][validation_index:]
        )


split_train_test_validation = TrainTestValidationSplitter(
    config.data.context,
    config.data.frac_test,
    config.data.frac_validate
)
