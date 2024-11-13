import torch as pt
from swak.misc import ArgRepr
from swak.pt.types import Module, Tensor
from ..config import config
from .data import TestData
from .trainer import loss
from .types import Validation


# ToDo: Compute "perplexity"!
class Validator(ArgRepr):

    def __init__(self, loss: Module, batch_size: int) -> None:
        super().__init__(loss, batch_size)
        self.loss = loss
        self.batch_size = batch_size

    @staticmethod
    def top(k: int, out: Tensor, target: Tensor) -> float:
        matches = out.topk(k, dim=1).indices == target.unsqueeze(1)
        return matches.sum(dim=(0, 1)) / target.shape[0]

    @staticmethod
    def stat(positions: Tensor) -> tuple[float, float]:
        return positions.mean().item(), 1.96 * positions.std().item()

    def __call__(self, model: Module, data: TestData) -> Validation:
        n = 0
        top_1 = 0
        top_2 = 0
        top_5 = 0
        val_loss = 0.0

        model.eval()
        with pt.no_grad():
            for features, target in data.sample(self.batch_size):
                batch_n = target.shape[0]

                (out,) = model(*features)

                batch_loss = self.loss(out, target).item()
                val_loss += batch_n * (batch_loss - val_loss) / (n + batch_n)

                batch_top_1 = self.top(1, out, target)
                top_1 += batch_n * (batch_top_1 - top_1) / (n + batch_n)

                batch_top_2 = self.top(2, out, target)
                top_2 += batch_n * (batch_top_2 - top_2) / (n + batch_n)

                batch_top_5 = self.top(5, out, target)
                top_5 += batch_n * (batch_top_5 - top_5) / (n + batch_n)

                n += batch_n

        return val_loss, self.stat(top_1), self.stat(top_2), self.stat(top_5)


validate = Validator(loss, config.train.batch_size)
