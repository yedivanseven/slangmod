from typing import Any
from swak.pt.types import Tensor, Module
from ..tokenizers import Algo
from .abc import NextToken


class Greedy(NextToken):
    """Simply pick the most probable token, one at a time.

    Parameters
    ----------
    tokenizer: Algo
        Fully configured ``Algo`` wrapper around a trained tokenizer.
    model: Module
        The trained PyTorch model to use for text generation.
    max_tokens: int, optional
        The maximum number of tokens to generate in case the end-of-sequence
        token is not predicted by the model first. Defaults to 256.

    """

    def __init__(
            self,
            tokenizer: Algo,
            model: Module,
            max_tokens: int = 256,
            **_: Any
    ) -> None:
        super().__init__(tokenizer, model, max_tokens)

    def next_token_from_logits(self, logits: Tensor) -> Tensor:
        """Take the argmax over the probabilities of permissible tokens.

        Parameters
        ----------
        logits: Tensor
            1-D PyTorch tensor with un-normalized probabilities over all
            permissible tokens in the vocabulary.

        Returns
        -------
        Tensor
            Int64 scalar with the argmax of `logits`.

        """
        return logits.argmax(dim=-1)
