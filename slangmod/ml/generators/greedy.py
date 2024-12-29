from typing import Any
from swak.pt.types import Tensor, Module
from ..tokenizers import Algo
from .abc import NextToken


class Greedy(NextToken):

    def __init__(
            self,
            tokenizer: Algo,
            model: Module,
            max_tokens: int = 1024,
            **_: Any
    ) -> None:
        super().__init__(tokenizer, model, max_tokens, 1)

    def next_token_from_logits(self, logits: Tensor) -> Tensor:
        return logits.argmax(dim=-1)
