import torch as pt
import torch.distributions as ptd
from swak.pt.types import Tensor
from .abc import Generator


class Random(Generator):

    def sample(self, src: Tensor, mask: Tensor) -> list[int]:
        answer = []
        self.model.eval()
        with pt.no_grad():
            for _ in range(self.max_tokens):
                (out,) = self.model(src, None, mask, False)
                next_token = ptd.Categorical(logits=out[0, :, -1]).sample()
                answer.append(next_token.item())
                if next_token.item() == self.eos_id:
                    break
                src = pt.cat([src[:, 1:], next_token.view(1, 1)], dim=-1)
                mask = pt.cat([mask[:, 1:], self.zero], dim=-1)
        return answer
