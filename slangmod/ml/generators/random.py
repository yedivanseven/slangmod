from typing import Any
from collections.abc import Callable
import torch.distributions as ptd
from swak.pt.types import Module, Tensor
from ..tokenizers import Algo
from .abc import NextToken


class Random(NextToken):
    def __init__(
            self,
            tokenizer: Algo,
            model: Module,
            style: Callable[[str], str],
            max_tokens: int = 1024,
            temperature: float = 1.0,
            **_: Any
    ) -> None:
        super().__init__(tokenizer, model, style, max_tokens)
        self.temperature = temperature


    def next_token_from_logits(self, logits: Tensor) -> Tensor:
        scaled = logits / self.temperature
        return ptd.Categorical(logits=scaled).sample()
