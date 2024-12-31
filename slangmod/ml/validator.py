import math
from tqdm import tqdm
import torch as pt
from swak.misc import ArgRepr
from swak.pt.types import Module, Tensor
from ..config import config
from .data import TestData
from .trainer import loss
from .types import Validation


class Validator(ArgRepr):

    def __init__(self, loss: Module, batch_size: int) -> None:
        super().__init__(loss, batch_size)
        self.loss = loss
        self.batch_size = batch_size

    @staticmethod
    def top(k: int, logits: Tensor, targets: Tensor) -> Tensor:
        matches = logits.topk(k, dim=1).indices == targets.unsqueeze(1)
        return matches.sum(dim=(0, 1)) / targets.size(0)

    @staticmethod
    def stats(positions: Tensor) -> tuple[float, float]:
        return positions.mean().item(), 1.96 * positions.std().item()

    def perplexity(self, logits: Tensor, targets: Tensor) -> float:
        # Remember the "reduction" configured with the loss.
        cached_reduction = self.loss.reduction
        # Set its "reduction" to "none" ...
        self.loss.reduction = 'none'
        # ... to get the loss per sequence and token.
        losses = self.loss(logits, targets)
        # Revert the "reduction" back to what it was before.
        self.loss.reduction = cached_reduction

        # Count the number of non-padding tokens per sequence.
        n_non_padding = (targets > 0).sum(dim=1)
        # Create a mask to drop invalid sequences with only padding tokens.
        mask = n_non_padding > 0
        # Average the loss over all non-padding tokens per valid sequence.
        mean_sequence_losses = losses.sum(dim=1)[mask] / n_non_padding[mask]
        # Return the perplexity averaged over all sequences in the batch.
        return mean_sequence_losses.exp().mean(dim=0).item()

    def __call__(self, model: Module, data: TestData) -> Validation:
        n = 0
        top_1 = 0
        top_2 = 0
        top_5 = 0
        val_loss = 0.0
        perplexity = 0.0
        n_batches = math.ceil(data.n / self.batch_size)

        model.eval()
        self.loss.eval()
        with pt.inference_mode():
            batches = data.sample(self.batch_size)
            progress = tqdm(batches, 'Validate', n_batches, False)
            for features, targets in progress:
                batch_n = targets.size(0)

                logits, *_ = model(*features)

                batch_loss = self.loss(logits, targets).item()
                val_loss += batch_n * (batch_loss - val_loss) / (n + batch_n)
                progress.set_postfix(loss=f'{batch_loss:4.2f}')

                batch_px = self.perplexity(logits, targets)
                perplexity += batch_n * (batch_px - perplexity) / (n + batch_n)

                batch_top_1 = self.top(1, logits, targets)
                top_1 += batch_n * (batch_top_1 - top_1) / (n + batch_n)

                batch_top_2 = self.top(2, logits, targets)
                top_2 += batch_n * (batch_top_2 - top_2) / (n + batch_n)

                batch_top_5 = self.top(5, logits, targets)
                top_5 += batch_n * (batch_top_5 - top_5) / (n + batch_n)

                n += batch_n

        return (
            val_loss,
            perplexity,
            self.stats(top_1),
            self.stats(top_2),
            self.stats(top_5)
        )


validate = Validator(loss, config.train.batch_size)
