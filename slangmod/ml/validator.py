import torch as pt
from swak.pt.types import Module
from ..config import config
from .data import TestData
from .trainer import loss


class Validator:

    def __init__(
            self,
            loss: Module,
            batch_size: int,
            max_n: int | None = None
    ) -> None:
        self.loss = loss
        self.batch_size = batch_size
        self.max_n = max_n

    def __call__(self, model: Module, data: TestData) -> float:
        max_n = data.n if self.max_n is None else min(data.n, self.max_n)
        n = 0
        val_loss = 0.0
        model.eval()
        with pt.no_grad():
            for features, target in data.sample(self.batch_size, max_n):
                n_new = target.shape[0]
                loss = self.loss(*model(*features), target).item()
                val_loss += n_new * (loss - val_loss) / (n + n_new)
                n += n_new
        return val_loss


validate = Validator(loss, config.batch_size ,config.max_n)
