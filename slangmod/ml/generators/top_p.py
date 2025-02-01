from typing import Any
import torch.distributions as ptd
import torch.nn.functional as ptnf
from swak.pt.types import Module, Tensor
from ..tokenizers import Algo
from .abc import NextToken


class TopP(NextToken):
    """Randomly draw the next token from the top fraction of probability.

    If the model is very sure about what the next token should be, then most
    of the probability mass will be concentrated on that token and the random
    draw will be from among very few tokens. If, in contrast, the model is not
    so sure, and the probability mass is more widely distributed, then the
    random draw will be from among many more candidate tokens.

    Parameters
    ----------
    tokenizer: Algo
        Fully configured ``Algo`` wrapper around a trained tokenizer.
    model: Module
        The trained PyTorch model to use for text generation.
    max_tokens: int, optional
        The maximum number of tokens to generate in case the end-of-sequence
        token is not predicted by the model first. Defaults to 256.
    p: float, optional
        Candidate tokens to draw from are chosen by ranking all in order of
        descending probability and taking as many as possible before the sum
        of their individual probabilities exceeds `p`. Default to 1.0, which
        results in a draw from a categorical distribution over *all*
        eligible tokens.
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
            p: float = 1.0,
            temperature: float = 1.0,
            **_: Any
    ) -> None:
        super().__init__(tokenizer, model, max_tokens)
        self.p = p
        self.temperature = temperature

    def __repr__(self) -> str:
        extras = f', p={self.p}, temperature={self.temperature})'
        return super().__repr__()[:-1] + extras

    def next_token_from_logits(self, logits: Tensor) -> Tensor:
        """Randomly draw the next token from among the most probable ones.

        Parameters
        ----------
        logits: Tensor
            1-D PyTorch tensor with un-normalized probabilities over all
            permissible tokens in the vocabulary.

        Returns
        -------
        Tensor
            Int64 scalar with the ID of the next token randomly chosen from
            the top candidates that together have a probability of `p`.

        """
        scaled = logits / self.temperature
        probas = ptnf.softmax(scaled, dim=-1).sort(dim=-1, descending=True)
        # Boolean mask starting with True until cumsum reaches p
        top_p = probas.values.cumsum(dim=-1) <= self.p
        # We need at least one element to draw
        top_p[0] = True
        # Sample only from the top-p candidates
        sample = ptd.Categorical(probas.values[top_p]).sample()
        # Return their actual index among the logits
        return probas.indices[sample]
