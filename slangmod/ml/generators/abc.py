from typing import Any
from collections.abc import Callable
from abc import ABC, abstractmethod
from functools import cached_property
import torch as pt
from swak.pt.types import Module, Tensor, Tensors2T
from ..tokenizers import Algo

type Logits = tuple[Tensor, int]


class Generator(ABC):

    def __init__(
            self,
            tokenizer: Algo,
            model: Module,
            style: Callable[[str], str],
            max_tokens: int = 1024,
            **_: Any
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.style = style
        self.max_tokens = max_tokens
        # Initialize model buffers with a sequence of maximum length
        src = pt.randint(0, self.vocab, [self.context], device=model.device)
        mask = pt.zeros(self.context, dtype=model.dtype, device=model.device)
        _ = self.model(src.unsqueeze(0), None, mask.unsqueeze(0), False)

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        template = '{}(tokenizer, model, {}, {})'
        return template.format(cls, self.style, self.max_tokens)

    @property
    def vocab(self) -> int:
        return self.tokenizer.vocab

    @property
    def eos_id(self) -> int:
        return self.tokenizer.eos_id

    @property
    def context(self) -> int:
        return self.model.positions.context

    @cached_property
    def zero(self) -> Tensor:
        return pt.zeros(1, 1, dtype=self.model.dtype, device=self.model.device)

    def __call__(self, prompt: str) -> tuple[str, bool]:
        encoded = self.tokenizer.encode(prompt)
        encoded.truncate(self.context, direction='left')

        more = encoded.ids[-1] == self.eos_id

        src = pt.tensor(encoded.ids, device=self.model.device).unsqueeze(0)

        mask = pt.zeros_like(
            src,
            dtype=self.model.dtype,
            device=self.model.device
        ).where(
            src == self.tokenizer.unk_id,
            1.0
        ).log()

        self.model.eval()
        answer = self.predict(src, mask, more)
        return self.tokenizer.decode(answer), answer[-1] == self.eos_id

    @abstractmethod
    def predict(self, src: Tensor, mask: Tensor, more: bool) -> list[int]:
        ...


class NextToken(Generator):

    def logits(self, src: Tensor, mask: Tensor, more: bool) -> Logits:
        with pt.no_grad():
            (out,) = self.model(src, None, mask, False)
        return out[0, self.eos_id + more:, -1].float(), self.eos_id + more

    def step(self, token: Tensor, src: Tensor, mask: Tensor) -> Tensors2T:
        mask = pt.cat([mask, self.zero], dim=-1)[:, -self.context:]
        src = pt.cat([src, token.view(1, 1)], dim=-1)[:, -self.context:]
        return src, mask

    def predict(self, src: Tensor, mask: Tensor, more: bool) -> list[int]:
        answer = []

        logits, offset = self.logits(src, mask, more)
        next_token = self.next_token_from_logits(logits) + offset
        answer.append(next_token.item())
        src, mask = self.step(next_token, src, mask)

        more = False if more else next_token.item() == self.eos_id

        for _ in range(1, self.max_tokens):
            logits, offset = self.logits(src, mask, more)
            next_token = self.next_token_from_logits(logits) + offset
            answer.append(next_token.item())
            if next_token.item() == self.eos_id:
                break
            src, mask = self.step(next_token, src, mask)
            more = False

        return answer

    @abstractmethod
    def next_token_from_logits(self, logits: Tensor) -> Tensor:
        ...
