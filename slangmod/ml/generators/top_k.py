from typing import Any
import torch.distributions as ptd
from swak.pt.types import Module, Tensor
from ..tokenizers import Algo
from .abc import NextToken


class TopK(NextToken):
    """Randomly draw the next token from among the `k` most likely ones.

    Parameters
    ----------
    tokenizer: Algo
        Fully configured ``Algo`` wrapper around a trained tokenizer.
    model: Module
        The trained PyTorch model to use for text generation.
    max_tokens: int, optional
        The maximum number of tokens to generate in case the end-of-sequence
        token is not predicted by the model first. Defaults to 256.
    k: int or float, optional
        If an integer number > 0, the next token is drawn from a categorical
        distribution over the `k` most probable of all eligible tokens.
        A floating point number from the interval (0.0, 1.0) is interpreted
        as a fraction of all eligible tokens. Default to ``None``, resulting
        in a random draw from a categorical distribution over *all* eligible
        tokens.
    temperature: float, optional
        Higher temperatures concentrate more probability mass onto the most
        likely tokens, while lower temperatures spread the probability mass
        out among all eligible tokens. Defaults to 1.0, which results in
        unmodified logits.

    """

    def __init__(
            self,
            tokenizer: Algo,
            model: Module,
            max_tokens: int = 256,
            k: float | None = None,
            temperature: float = 1.0,
            **_: Any
    ) -> None:
        super().__init__(tokenizer, model, max_tokens)
        if k is None:
            self.k = self.max_k
        elif 0.0 < k < 1.0:
            self.k = round(k * self.max_k)
        elif k > 1.0:
            self.k = round(min(k, self.max_k))
        else:
            msg = f'k must be a number between 0.0 and {self.max_k}'
            raise ValueError(msg)
        self.temperature = temperature

    def __repr__(self) -> str:
        extras = f', k={self.k}, temperature={self.temperature})'
        return super().__repr__()[:-1] + extras

    @property
    def max_k(self) -> int:
        """Maximum permissible value for k."""
        return self.vocab - self.eos_id

    def next_token_from_logits(self, logits: Tensor) -> Tensor:
        """Randomly draw the next token from the top-k most probable ones.

        Parameters
        ----------
        logits: Tensor
            1-D PyTorch tensor with un-normalized probabilities over all
            permissible tokens in the vocabulary.

        Returns
        -------
        Tensor
            Int64 scalar with the ID of the next token randomly chosen from
            top-k candidates in `logits`.

        """
        top_k = logits.topk(min(self.k, logits.size(-1)), dim=-1)
        scaled = top_k.values / self.temperature
        sample = ptd.Categorical(logits=scaled).sample()
        return top_k.indices[sample]
