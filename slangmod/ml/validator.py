import torch as pt
from swak.pt.types import Module
from ..config import config
from .data import TestData
from .trainer import loss


class Validator:

    def __init__(self, loss: Module, batch_size: int) -> None:
        self.loss = loss
        self.batch_size = batch_size

    def __call__(self, model: Module, data: TestData) -> float:
        n = 0
        val_loss = 0.0
        model.eval()
        with pt.no_grad():
            for features, target in data.sample(self.batch_size):
                n_new = target.shape[0]
                loss = self.loss(*model(*features), target).item()
                val_loss += n_new * (loss - val_loss) / (n + n_new)
                n += n_new
        return val_loss
        # ToDo: Add batch-wise, accumulated accuracy (top-1, top-2, top-5)


validate = Validator(loss, config.batch_size)
