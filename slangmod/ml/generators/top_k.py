from typing import Any
import torch.distributions as ptd
from swak.pt.types import Module, Tensor
from ..tokenizers import Algo
from .abc import NextToken


class TopK(NextToken):

    def __init__(
            self,
            tokenizer: Algo,
            model: Module,
            max_tokens: int = 1024,
            k: float | None = None,
            temperature: float = 1.0,
            **_: Any
    ) -> None:
        super().__init__(tokenizer, model, max_tokens, 1)
        if k is None:
            self.k = self.max_k
        elif k < 1.0:
            self.k = round(max(1.0, k * self.max_k))
        else:
            self.k = round(min(k, self.max_k))
        self.temperature = temperature

    def __repr__(self) -> str:
        extras = f', k={self.k}, temperature={self.temperature})'
        return super().__repr__()[:-1] + extras

    @property
    def max_k(self) -> int:
        return self.vocab - self.eos_id

    def next_token_from_logits(self, logits: Tensor) -> Tensor:
        top_k = logits.topk(self.k, dim=-1)
        scaled = top_k.values / self.temperature
        sample = ptd.Categorical(logits=scaled).sample()
        return top_k.indices[sample]
