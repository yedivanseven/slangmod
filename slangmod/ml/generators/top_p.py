from typing import Any
from collections.abc import Callable
import torch.distributions as ptd
import torch.nn.functional as ptnf
from swak.pt.types import Module, Tensor
from ..tokenizers import Algo
from .abc import NextToken


class TopP(NextToken):

    def __init__(
            self,
            tokenizer: Algo,
            model: Module,
            style: Callable[[str], str],
            max_tokens: int = 1024,
            p: float = 1.0,
            temperature: float = 1.0,
            **_: Any
    ) -> None:
        super().__init__(tokenizer, model, style, max_tokens)
        self.p = p
        self.temperature = temperature

    def __repr__(self) -> str:
        return super().__repr__()[:-1] + f', p={self.p})'

    def next_token_from_logits(self, logits: Tensor) -> Tensor:
        scaled = logits / self.temperature
        probas = ptnf.softmax(scaled, dim=-1).sort(dim=-1, descending=True)
        top_p = probas.values.cumsum(dim=-1) <= self.p
        top_p[0] = True  # We need at least one element to draw!
        sample = ptd.Categorical(probas.values[top_p]).sample()
        return probas.indices[sample]
