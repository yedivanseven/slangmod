from typing import Any
from collections.abc import Callable
import torch as pt
import torch.distributions as ptd
from tokenizers import Tokenizer
from swak.pt.types import Module, Tensor
from .abc import Generator


class TopK(Generator):

    def __init__(
            self,
            tokenizer: Tokenizer,
            model: Module,
            wrap: Callable[[str], str],
            max_tokens: int = 1024,
            unk_id: int = 1,
            eos_id: int = 2,
            k: float | None = None,
            **_: Any
    ) -> None:
        super().__init__(tokenizer, model, wrap, max_tokens, unk_id, eos_id)
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
        return self.tokenizer.get_vocab_size()

    def sample(self, src: Tensor, mask: Tensor) -> list[int]:
        answer = []
        self.model.eval()
        with pt.no_grad():
            for _ in range(self.max_tokens):
                (out,) = self.model(src, None, mask, False)
                top_k = out[0, :, -1].topk(self.k, dim=-1)
                sample = ptd.Categorical(logits=top_k.values).sample()
                next_token = top_k.indices[sample]
                answer.append(next_token.item())
                if next_token.item() == self.eos_id:
                    break
                src = pt.cat([src[:, 1:], next_token.view(1, 1)], dim=-1)
                mask = pt.cat([mask[:, 1:], self.zero], dim=-1)
        return answer
