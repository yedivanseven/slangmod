from typing import Any
import torch.distributions as ptd
from swak.pt.types import Module, Tensor
from ..tokenizers import Algo
from .abc import NextToken


class Random(NextToken):

    def __init__(
            self,
            tokenizer: Algo,
            model: Module,
            max_tokens: int = 1024,
            temperature: float = 1.0,
            **_: Any
    ) -> None:
        super().__init__(tokenizer, model, max_tokens, 1)
        self.temperature = temperature

    def __repr__(self) -> str:
        extras = f', temperature={self.temperature})'
        return super().__repr__()[:-1] + extras

    def next_token_from_logits(self, logits: Tensor) -> Tensor:
        scaled = logits / self.temperature
        return ptd.Categorical(logits=scaled).sample()
