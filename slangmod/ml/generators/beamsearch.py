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
            penalty: float = 0.8,
            **_: Any
    ) -> None:
        super().__init__(tokenizer, model, style, max_tokens)
        self.width = width
        self.penalty = penalty

    def predict(self, src: Tensor, mask: Tensor) -> list[int]:

        with pt.no_grad():
            (out,) = self.model(src, None, mask, False)
        logits = out[0, self.eos_id + 1:, -1].float()
        top_k = ptnf.log_softmax(logits, dim=-1).topk(self.width, dim=-1)

        all_seqs = top_k.indices.view(-1, 1) + self.eos_id + 1
        all_prob = top_k.values
        all_size = pt.ones_like(all_prob).long()
        all_vals = top_k.values

        for _ in range(self.max_tokens - 1):

            eos = all_seqs[:, -1] == self.eos_id

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
            logits = out[:, self.eos_id:, -1].float()
            top_k = ptnf.log_softmax(logits, dim=-1).topk(self.width, dim=-1)
            next_tokens = top_k.indices.view(-1, 1) + self.eos_id
            log_prob = top_k.values.ravel()

            grow_seqs = all_seqs[~eos].repeat_interleave(self.width, dim=0)
            grow_seqs = pt.cat([grow_seqs, next_tokens], dim=-1)
            grow_prob = all_prob[~eos].repeat_interleave(self.width) + log_prob
            grow_size = all_size[~eos].repeat_interleave(self.width) + 1
            grow_vals = grow_prob / grow_size**self.penalty

            all_seqs = pt.cat([eos_seqs, grow_seqs], dim=0)
            all_prob = pt.cat([eos_prob, grow_prob], dim=0)
            all_size = pt.cat([eos_size, grow_size], dim=0)
            all_vals = pt.cat([eos_vals, grow_vals], dim=0)

            _, top_k_indices = all_vals.topk(self.width, dim=-1)

            all_seqs = all_seqs[top_k_indices]
            all_prob = all_prob[top_k_indices]
            all_size = all_size[top_k_indices]
            all_vals = all_vals[top_k_indices]

        winner = all_vals.argmax()
        return all_seqs[winner].tolist()