from typing import Any
from collections.abc import Callable
import torch.distributions as ptd
from swak.pt.types import Module, Tensor
from ..tokenizers import Algo
from .abc import NextToken


class TopK(NextToken):

    def __init__(
            self,
            tokenizer: Algo,
            model: Module,
            style: Callable[[str], str],
            max_tokens: int = 1024,
            k: float | None = None,
            temperature: float = 1.0,
            **_: Any
    ) -> None:
        super().__init__(tokenizer, model, style, max_tokens)
        if k is None:
            self.k = self.max_k
        elif k < 1.0:
            self.k = round(max(1.0, k * self.max_k))
        else:
            self.k = round(min(k, self.max_k))
        self.temperature = temperature

    def __repr__(self) -> str:
        return super().__repr__()[:-1] + f', k={self.k})'

    @property
    def max_k(self) -> int:
        return self.vocab - self.eos_id

    def next_token_from_logits(self, logits: Tensor) -> Tensor:
        top_k = logits.topk(self.k, dim=-1)
        scaled = top_k.values / self.temperature
        sample = ptd.Categorical(logits=scaled).sample()
        return top_k.indices[sample]
