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

    def __call__(self, prompt: str) -> str:
        encoded = self.tokenizer.encode(self.style(prompt))
        encoded.truncate(self.context, direction='left')

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
        return self.tokenizer.decode(self.predict(src, mask))

    @abstractmethod
    def predict(self, src: Tensor, mask: Tensor) -> list[int]:
        ...


class NextToken(Generator):

    def logits(self, src: Tensor, mask: Tensor, first: bool = False) -> Logits:
        with pt.no_grad():
            (out,) = self.model(src, None, mask, False)
        return out[0, self.eos_id + first:, -1].float(), self.eos_id + first

    def step(self, token: Tensor, src: Tensor, mask: Tensor) -> Tensors2T:
        mask = pt.cat([mask, self.zero], dim=-1)[:, -self.context:]
        src = pt.cat([src, token.view(1, 1)], dim=-1)[:, -self.context:]
        return src, mask

    @abstractmethod
    def predict(self, src: Tensor, mask: Tensor) -> list[int]:
        ...
