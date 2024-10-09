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
        self.__perm = None

    @property
    def train(self) -> float:
        return 1.0 - self.test - self.validation

    def __call__(self, data: Tensor) -> Tensors3T:
        n = data.shape[0]
        self.__perm = pt.randperm(n) if self.__perm is None else self.__perm
        test_index = int(n * self.train)
        validation_index = int(n * (self.train + self.test))
        return (
            data[self.__perm][:test_index + self.context],
            data[self.__perm][test_index + 1:validation_index + self.context],
            data[self.__perm][validation_index + 1:]
        )


split_train_test_validation = TrainTestValidationSplitter(
    config.context,
    config.frac_test,
    config.frac_validate
)
