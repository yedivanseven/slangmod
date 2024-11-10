from swak.pt.types import Tensor
from .abc import NextToken


class Greedy(NextToken):

    def next_token_from_logits(self, logits: Tensor) -> Tensor:
        return logits.argmax(dim=-1)
