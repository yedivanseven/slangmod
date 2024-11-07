import torch.distributions as ptd
from swak.pt.types import Tensor
from .abc import NextToken


class Random(NextToken):

    def predict(self, src: Tensor, mask: Tensor) -> list[int]:
        answer = []

        logits, offset = self.logits(src, mask, first=True)
        next_token = ptd.Categorical(logits=logits).sample() + offset
        answer.append(next_token.item())
        src, mask = self.step(next_token, src, mask)

        for _ in range(self.max_tokens - 1):
            logits, offset = self.logits(src, mask, first=False)
            next_token = ptd.Categorical(logits=logits).sample() + offset
            answer.append(next_token.item())
            if next_token.item() == self.eos_id:
                break
            src, mask = self.step(next_token, src, mask)

        return answer
