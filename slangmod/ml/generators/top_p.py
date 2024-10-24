from typing import Any
from collections.abc import Callable
import torch as pt
import torch.nn as ptn
import torch.distributions as ptd
from tokenizers import Tokenizer
from swak.pt.types import Module, Tensor
from .abc import Generator


class TopP(Generator):

    def __init__(
            self,
            tokenizer: Tokenizer,
            model: Module,
            wrap: Callable[[str], str],
            max_tokens: int = 1024,
            unk_id: int = 1,
            eos_id: int = 2,
            p: float = 1.0,
            **_: Any
    ) -> None:
        super().__init__(tokenizer, model, wrap, max_tokens, unk_id, eos_id)
        self.p = p

    def __repr__(self) -> str:
        return super().__repr__()[:-1] + f', p={self.p})'

    def sample(self, src: Tensor, mask: Tensor) -> list[int]:
        answer = []
        self.model.eval()
        with pt.no_grad():
            for _ in range(self.max_tokens):
                (out,) = self.model(src, None, mask, None)
                probas = ptn.Softmax(-1)(out[0, :, -1]).sort(descending=True)
                top_p = probas.values.cumsum(dim=-1) <= self.p
                top_p[0] = True  # We need at least one element to draw!
                sample = ptd.Categorical(probas.values[top_p]).sample()
                next_token = probas.indices[sample]
                answer.append(next_token.item())
                if next_token.item() == self.eos_id:
                    break
                src = pt.cat([src[:, 1:], next_token.view(1, 1)], dim=-1)
                mask = pt.cat([mask[:, 1:], self.zero], dim=-1)
        return answer
