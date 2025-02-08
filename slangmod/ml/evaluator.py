import math
from tqdm import tqdm
import torch as pt
from torch.nn import CrossEntropyLoss
from swak.misc import ArgRepr
from swak.pt.types import Module, Tensor
from ..config import config
from .data import TestData
from .trainer import criterion
from .types import Evaluation

__all__ = [
    'Evaluator',
    'evaluate_model'
]


class Evaluator(ArgRepr):
    """Compute statistics on hold-out data to evaluate model performance.

    Parameters
    ----------
    loss: Module
        An instance of `CrossEntropyLoss <https://pytorch.org/docs/stable/
        generated/torch.nn.CrossEntropyLoss.html#crossentropyloss>`_ with the
        exact same parameters that were used to train your model with
        the possible exception of `label_smoothing` which may be set to 0.0
    batch_size: int
        The batch size to request from the data producer when computing
        evaluation metrics.
    show_progress: bool, optional
        Whether to show a progress bar that provides visual feedback in the
        console during the validation process. Defaults to ``True``.

    Raises
    ------
    ValueError
        If the "reduction" of the `loss` is not "mean".

    """

    def __init__(
            self,
            loss: CrossEntropyLoss,
            batch_size: int,
            show_progress: bool = True
    ) -> None:
        super().__init__(loss, batch_size, show_progress)
        self.loss = self.__sane(loss)
        self.batch_size = batch_size
        self.show_progress = show_progress

    @staticmethod
    def __sane(loss: CrossEntropyLoss) -> CrossEntropyLoss:
        """Check that the "reduction" of the CrossEntropyLoss is "mean"."""
        if loss.reduction != 'mean':
            tmp = ('The "reduction" of the CrossEntropyLoss'
                   ' must be "mean" and not "{}"!')
            msg = tmp.format(loss.reduction)
            raise ValueError(msg)
        return loss

    @property
    def pad_id(self) -> int:
        """Index of the padding token"""
        return self.loss.ignore_index

    def top(self, k: int, logits: Tensor, targets: Tensor) -> Tensor:
        """Top-k correct predictions summed over all non-padding tokens.

        Parameters
        ----------
        k: int
            The target token index has to be within the top `k` most probable
            indices predicted by the model, provided it is not padding.
        logits: Tensor
            The logits predicted by the model. Must be of sizes (`batch_size`,
            `vocab`, `S`), where `S` is the (padded) sequence length.
        targets: Tensor
            The true indices of the tokens the model should predict. Must be
            of sizes (`batch_size`, `S`).

        Returns
        -------
        Tensor
            Count of correct top-k predictions over all non-padding tokens.

        """
        # Matches with target tokens have sizes (batch_size, k, seq_len).
        matches = logits.topk(k, dim=1).indices == targets.unsqueeze(1)
        # Mask of sizes (batch, 1, seq_len) to select all non-padding tokens.
        is_non_pad_token = targets.unsqueeze(1) != self.pad_id
        # Return the number of matches with non-padding target tokens.
        return (matches * is_non_pad_token).sum()

    def perplexity(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute the perplexity summed over a minibatch of sequences.

        Parameters
        ----------
        logits: Tensor
            The logits predicted by the model. Must be of sizes (`batch_size`,
            `vocab`, `S`), where `S` is the (padded) sequence length.
        targets: Tensor
            The true indices of the tokens the model should predict. Must be
            of sizes (`batch_size`, `S`).

        Returns
        -------
        Tensor
            The perplexity summed over all sequences in the minibatch.

        Note
        ----
        Sequences consisting of only padding tokens are not expected and will
        lead to a division by zero.

        """
        # Set the "reduction" of the loss to "none" ...
        self.loss.reduction = 'none'
        # ... to get the loss per sequence and token (zero for padding tokens).
        losses = self.loss(logits, targets)
        # Revert the "reduction" back "mean".
        self.loss.reduction = 'mean'
        # Count the number of non-padding tokens per sequence.
        n_non_pad_tokens = (targets != self.pad_id).sum(dim=1)
        # Average the loss over all non-padding tokens for each sequence.
        mean_sequence_losses = losses.sum(dim=1) / n_non_pad_tokens
        # Return the perplexity summed over all sequences in the batch.
        return mean_sequence_losses.exp().sum(dim=0)

    def __call__(self, model: Module, data: TestData) -> Evaluation:
        """Compute metrics on validation data to evaluate model performance.

        Parameters
        ----------
        model: Module
            The model to evaluate.
        data: TestData
            The hold-out validation data to evaluate the model on

        Returns
        -------
        loss: float
            The loss averaged over all non-padding tokens.
        perplexity: float
            The perplexity averaged over all sequences.
        accuracy: float
            The fraction of non-padding tokens predicted correctly.
        top_2: float
            The fraction of correct non-padding tokens within the top-2
            most probable tokens predicted by the model.
        top_5: float
            The fraction of correct non-padding tokens within the top-5
            most probable tokens predicted by the model.

        """
        # Initialize counters and accumulation variables
        ema = None
        n_tok = pt.tensor(0, device=data.device, dtype=pt.long)
        n_seq = pt.tensor(0, device=data.device, dtype=pt.long)
        top_1 = pt.tensor(0.0, device=data.device, dtype=pt.float)
        top_2 = pt.tensor(0.0, device=data.device, dtype=pt.float)
        top_5 = pt.tensor(0.0, device=data.device, dtype=pt.float)
        val_loss = pt.tensor(0.0, device=data.device, dtype=pt.double)
        perplexity = pt.tensor(0.0, device=data.device, dtype=pt.float)
        # Initialize iterator over the validation data
        n_batches = math.ceil(data.n / self.batch_size)
        progress = tqdm(
            data.sample(self.batch_size),
            desc='Validate',
            total=n_batches,
            leave=False,
            disable=not self.show_progress
        )
        # Set model (and loss) into evaluation mode.
        model.eval()
        self.loss.eval()
        with pt.inference_mode():
            # Targets are of sizes (batch_size, seq_len)
            for features, targets in progress:
                # Mask of same size to select all non-padding tokens.
                valid_tokens = (targets != self.pad_id)
                # Count the total number of non-padding tokens in the batch.
                batch_n_tok = valid_tokens.sum()
                # Mask to select sequences with at least 1 non-padding token.
                valid_seqs = valid_tokens.sum(dim=-1) > 0
                # Count the number of sequences with >=1 non-padding token ...
                batch_n_seq = valid_seqs.sum()
                # ... and move on to the next batch if there are none.
                if batch_n_seq == 0:
                    continue
                # Logits are of dimension (batch_size, vocab_size, seq_len)
                logits, *_ = model(*features)
                # Retain logits and targets only for valid sequences
                valid_logits = logits[valid_seqs]
                valid_targets = targets[valid_seqs]

                # Loss averaged over all non-padding tokens in the batch.
                batch_loss = self.loss(valid_logits, valid_targets)
                val_loss += (
                    batch_n_tok * (batch_loss - val_loss) /
                    (n_tok + batch_n_tok)
                )
                # Exponential moving average for reporting in progress bar.
                ema = batch_loss if ema is None else 0.5 * (ema + batch_loss)
                progress.set_postfix(loss=f'{ema:4.2f}')

                # Perplexity summed over all sequences in the batch.
                perplexity += self.perplexity(valid_logits, valid_targets)
                # Accuracy averaged over all non-padding tokens in the batch.
                top_1 += self.top(1, valid_logits, valid_targets)
                # Top-2 accuracy averaged over all non-padding tokens.
                top_2 += self.top(2, valid_logits, valid_targets)
                # Top-5 accuracy averaged over all non-padding tokens.
                top_5 += self.top(5, valid_logits, valid_targets)

                # Increment counters for total numbers of tokens and sequences.
                n_tok += batch_n_tok
                n_seq += batch_n_seq

        return (
            val_loss.item(),
            (perplexity / n_seq).item(),
            (top_1 / n_tok).item(),
            (top_2 / n_tok).item(),
            (top_5 / n_tok).item()
        )


# Provide a ready-to-use instance of the Evaluator
evaluate_model = Evaluator(
    loss=criterion,
    batch_size=config.train.batch_size,
    show_progress=config.progress
)
