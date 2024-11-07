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
            **_: Any
    ) -> None:
        super().__init__(tokenizer, model, style, max_tokens)
        if k is None:
            self.k = self.max_k
        elif k < 1.0:
            self.k = round(max(1.0, k * self.max_k))
        else:
            self.k = round(k)

    def __repr__(self) -> str:
        return super().__repr__()[:-1] + f', k={self.k})'

    @property
    def max_k(self) -> int:
        return self.vocab - self.eos_id

    def predict(self, src: Tensor, mask: Tensor) -> list[int]:
        answer = []

        logits, offset = self.logits(src, mask, first=True)
        top_k = logits.topk(self.k, dim=-1)
        sample = ptd.Categorical(logits=top_k.values).sample()
        next_token = top_k.indices[sample] + offset
        answer.append(next_token.item())
        src, mask = self.step(next_token, src, mask)

        for _ in range(self.max_tokens - 1):
            logits, offset = self.logits(src, mask, first=False)
            top_k = logits.topk(self.k, dim=-1)
            sample = ptd.Categorical(logits=top_k.values).sample()
            next_token = top_k.indices[sample] + offset
            answer.append(next_token.item())
            if next_token.item() == self.eos_id:
                break
            src, mask = self.step(next_token, src, mask)

        return answer
