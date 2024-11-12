from typing import Any
from collections.abc import Callable
import torch as pt
import torch.nn.functional as ptnf
from swak.pt.types import Module, Tensor
from ..tokenizers import Algo
from .abc import Generator


class BeamSearch(Generator):

    def __init__(
            self,
            tokenizer: Algo,
            model: Module,
            style: Callable[[str], str],
            max_tokens: int = 1024,
            width: int = 10,
            boost: float = 0.8,
            **_: Any
    ) -> None:
        super().__init__(tokenizer, model, style, max_tokens)
        self.width = width
        self.boost = boost

    def __repr__(self) -> str:
        prefix = super().__repr__()[:-1]
        suffix = f', width={self.width}, boost={self.boost})'
        return prefix + suffix

    def predict(self, src: Tensor, mask: Tensor, more: bool) -> list[int]:

        with pt.no_grad():
            (out,) = self.model(src, None, mask, False)
        logits = out[0, self.eos_id + more:, -1].float()
        top_k = ptnf.log_softmax(logits, dim=-1).topk(self.width, dim=-1)

        all_seqs = top_k.indices.view(-1, 1) + self.eos_id + more
        all_prob = top_k.values
        all_size = pt.ones_like(all_prob).long()
        all_vals = top_k.values

        more = False if more else (all_seqs == self.eos_id).any()
        eos = pt.zeros(all_seqs.shape[0], dtype=pt.bool)

        for _ in range(1, self.max_tokens):

            eos = eos if more else all_seqs[:, -1] == self.eos_id

            if eos.all():
                break

            eos_seqs = ptnf.pad(all_seqs[eos], (0, 1), value=self.eos_id)
            eos_prob = all_prob[eos]
            eos_size = all_size[eos]
            eos_vals = all_vals[eos]
            n_remain = all_seqs[~eos].shape[0]

            inp = pt.cat([src.expand(n_remain, -1), all_seqs[~eos]], dim=-1)
            mask = pt.cat([mask, self.zero], dim=-1)[:, -self.context:]
            unk = mask.expand(n_remain, -1)

            with pt.no_grad():
                (out,) = self.model(inp[:, -self.context:], None, unk, False)
            logits = out[:, self.eos_id + more:, -1].float()
            top_k = ptnf.log_softmax(logits, dim=-1).topk(self.width, dim=-1)
            next_tokens = top_k.indices.view(-1, 1) + self.eos_id + more
            log_prob = top_k.values.ravel()

            grow_seqs = all_seqs[~eos].repeat_interleave(self.width, dim=0)
            grow_seqs = pt.cat([grow_seqs, next_tokens], dim=-1)
            grow_prob = all_prob[~eos].repeat_interleave(self.width) + log_prob
            grow_size = all_size[~eos].repeat_interleave(self.width) + 1
            grow_vals = grow_prob / grow_size**self.boost

            all_seqs = pt.cat([eos_seqs, grow_seqs], dim=0)
            all_prob = pt.cat([eos_prob, grow_prob], dim=0)
            all_size = pt.cat([eos_size, grow_size], dim=0)
            all_vals = pt.cat([eos_vals, grow_vals], dim=0)

            _, top_k_indices = all_vals.topk(self.width, dim=-1)

            all_seqs = all_seqs[top_k_indices]
            all_prob = all_prob[top_k_indices]
            all_size = all_size[top_k_indices]
            all_vals = all_vals[top_k_indices]

            more = False

        winner = all_vals.argmax()
        answer = all_seqs[winner].tolist()
        if self.eos_id in answer[1:]:
            eos_pos = answer[1:].index(2) + 2
        else:
            eos_pos = len(answer)
        return answer[:eos_pos]
