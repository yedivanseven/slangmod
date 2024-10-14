import torch as pt
from swak.pt.types import Module
from ..config import config
from .data import TestData
from .trainer import loss


class Validator:

    def __init__(self, loss: Module, max_n: int | None = None) -> None:
        self.loss = loss
        self.max_n = max_n

    def __call__(self, model: Module, data: TestData):
        max_n = data.n if self.max_n is None else min(data.n, self.max_n)
        src, tgt = data.sample(max_n)

        model.eval()
        with pt.no_grad():
            (prediction,) = model(*src)

        validation_loss = self.loss(prediction, tgt).item()

        top1 = prediction[:, :, -1].topk(1, -1).indices
        top2 = prediction[:, :, -1].topk(2, -1).indices
        top5 = prediction[:, :, -1].topk(5, -1).indices

        acc1 = (tgt[:, -1:].expand_as(top1) == top1).float().mean().item()
        acc2 = (tgt[:, -1:].expand_as(top2) == top2).float().mean().item()
        acc5 = (tgt[:, -1:].expand_as(top5) == top5).float().mean().item()

        return validation_loss, acc1, acc2, acc5


validate = Validator(loss, config.max_n)
