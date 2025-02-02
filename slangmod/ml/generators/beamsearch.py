from typing import Any
import torch as pt
import torch.nn.functional as ptnf
from swak.pt.types import Module, Tensor
from ..tokenizers import Algo
from .abc import Generator


class BeamSearch(Generator):
    """Perform a beam search for the most likely sequence of predicted tokens.

    Parameters
    ----------
    tokenizer: Algo
        Fully configured ``Algo`` wrapper around a trained tokenizer.
    model: Module
        The trained PyTorch model to use for text generation.
    max_tokens: int, optional
        The maximum number of tokens to generate in case the end-of-sequence
        token is not predicted by the model first. Defaults to 256.
    width: int, optional
        The width of the beam search. Defaults to 4.
    boost: float, optional
        Boost the length of the generated answers. The *higher* this number,
        the *longer* the answers. Conversely, *lower* numbers promote *shorter*
        answers. Defaults to 1.0, which ranks answers purely on their
        (log-)probabilities.

    """

    def __init__(
            self,
            tokenizer: Algo,
            model: Module,
            max_tokens: int = 256,
            width: int = 4,
            boost: float = 1.0,
            **_: Any
    ) -> None:
        super().__init__(tokenizer, model, max_tokens, width)
        self.boost = boost

    def __repr__(self) -> str:
        extras = f', width={self.width}, boost={self.boost})'
        return super().__repr__()[:-1] + extras

    def predict(self, src: Tensor, mask: Tensor, more: bool) -> list[int]:
        """Beam search for the most likely sequence of predicted tokens.

        Parameters
        ----------
        src: Tensor
            PyTorch tensor with the `prompt` converted to integer token
            IDs of shape (1, `S`) with `S` being the number of tokens.
        mask: Tensor
            Additive attention mask of the same shape with `-inf` indicating
            positions that should *not* be attended to and 0 that they should.
        more: bool
            If ``True``, the input to the model ends with and end-of-sequence
            (EOS) token and, therefore, the model must predict at least one
            token that is *not* EOS. If ``False``, the model may predict EOS
            first, but must follow that up with at least one non-EOS token.

        Returns
        -------
        list
            Integer token IDs of the model response.

        """
        # Predict the first token of the answer ...
        with pt.inference_mode():
            out, *_ = self.model(src, None, mask, False)
        # ... with or without EOS (depending on "more").
        logits = out[0, self.eos_id + more:self.vocab, -1].float()
        # We need the log-probabilities of the top "width" tokens.
        top_k = ptnf.log_softmax(logits, dim=-1).topk(self.width, dim=-1)

        # Reshape to (width, 1) and convert back to token index in the vocab,
        # thus initializing the buffer for _all_ candidate sequences.
        all_seqs = top_k.indices.view(-1, 1) + self.eos_id + more
        all_prob = top_k.values
        all_size = pt.ones_like(all_prob).long()
        all_vals = top_k.values

        # If we restricted the first token to not be EOS, the next one can be.
        # If we did not, and it is, we need at least one more non-EOS token.
        more = False if more else (all_seqs == self.eos_id).any().item()

        # Initialize a 1-D boolean mask indicating sequences already ended.
        eos = pt.zeros(all_seqs.size(0), dtype=pt.bool)

        for _ in range(1, self.max_tokens):
            # Which sequences already have an EOS token as their last token?
            eos = eos if more else all_seqs[:, -1] == self.eos_id
            # If it's all of them, we're done.
            if eos.all():
                break

            # If not, append ended sequences with another EOS token ...
            eos_seqs = ptnf.pad(all_seqs[eos], (0, 1), value=self.eos_id)
            # ... and set aside their log-probabilities, lengths, and scores.
            eos_prob = all_prob[eos]
            eos_size = all_size[eos]
            eos_vals = all_vals[eos]

            # How many candidates remain?
            n_remain = all_seqs[~eos].size(0)
            # Repeat original input accordingly and append unfinished sequences
            inp = pt.cat([src.expand(n_remain, -1), all_seqs[~eos]], dim=-1)
            # Extend the mask by one 0 (we do not predict unknown tokens).
            mask = pt.cat([mask, self.zero], dim=-1)[:, -self.context:]

            # Get next-token probabilities for all remaining sequences.
            with pt.inference_mode():
                out, *_ = self.model(inp[:, -self.context:], None, mask, False)
            logits = out[:, self.eos_id + more:self.vocab, -1].float()

            # Get top "width" most likely tokens for all remaining sequences...
            top_k = ptnf.log_softmax(logits, dim=-1).topk(self.width, dim=-1)
            # ... convert their indices back to token indices in the vocab ...
            next_tokens = top_k.indices.view(-1, 1) + self.eos_id + more
            # ... and get their log_probabilities in 1-D shape.
            log_prob = top_k.values.ravel()

            # Repeat each remaining sequence "width" times, ...
            grow_seqs = all_seqs[~eos].repeat_interleave(self.width, dim=0)
            # ... append the "width" most likely tokens, ...
            grow_seqs = pt.cat([grow_seqs, next_tokens], dim=-1)
            # ... add their log-probabilities, ...
            grow_prob = all_prob[~eos].repeat_interleave(self.width) + log_prob
            # ... increase their length by one, ...
            grow_size = all_size[~eos].repeat_interleave(self.width) + 1
            # ... and re-evaluate their score.
            grow_vals = grow_prob / grow_size**self.boost

            # Re-unite the new candidates with the already ended sequences, ...
            all_seqs = pt.cat([eos_seqs, grow_seqs], dim=0)
            all_prob = pt.cat([eos_prob, grow_prob], dim=0)
            all_size = pt.cat([eos_size, grow_size], dim=0)
            all_vals = pt.cat([eos_vals, grow_vals], dim=0)
            # ... get the "width" highest-scoring ones, ...
            top_k = all_vals.topk(self.width, dim=-1)
            # ... and keep only those in the candidate list for the next round.
            all_seqs = all_seqs[top_k.indices]
            all_prob = all_prob[top_k.indices]
            all_size = all_size[top_k.indices]
            all_vals = all_vals[top_k.indices]

            more = False  # From now on, EOS is acceptable in any case.

        winner = all_vals.argmax()
        answer = all_seqs[winner].tolist()
        # Sequence may have an EOS in the beginning (depending on the initial
        # "more", and shorter sequences potentially have many EOS at the end:
        if self.eos_id in answer[1:]:
            eos_pos = answer[1:].index(self.eos_id) + 2  # This keeps one EOS.
        else:
            eos_pos = len(answer)
        return answer[:eos_pos]  # Answer with at most one EOS at the end.
