from typing import Any
from collections.abc import Callable
from abc import ABC, abstractmethod
import torch as pt
from tokenizers import Tokenizer
from swak.pt.types import Module, Tensor, Device, Dtype


class Generator(ABC):

    def __init__(
            self,
            tokenizer: Tokenizer,
            model: Module,
            wrap: Callable[[str], str],
            max_tokens: int = 1024,
            unk_id: int = 1,
            eos_id: int = 2,
            **_: Any
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.wrap = wrap
        self.max_tokens = max_tokens
        self.unk_id = unk_id
        self.eos_id = eos_id

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(..., {self.wrap})'

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    @property
    def context(self) -> int:
        return self.model.context

    @property
    def device(self) -> Device:
        return self.model.device

    @property
    def dtype(self) -> Dtype:
        return self.model.dtype

    @property
    def zero(self) -> Tensor:
        return pt.zeros(1, 1, dtype=self.dtype, device=self.device)

    def __call__(self, prompt: str) -> str:
        encoded = self.tokenizer.encode(self.wrap(prompt))
        encoded.truncate(self.context, direction='left')
        encoded.pad(self.context, direction='left')

        src = pt.tensor(encoded.ids, device=self.device).unsqueeze(0)

        padding_mask = pt.tensor(
            encoded.attention_mask,
            dtype=self.model.dtype,
            device=self.model.device
        ).log(
        ).unsqueeze(0)
        unknown_mask = pt.zeros_like(
            src,
            dtype=self.model.dtype,
            device=self.model.device
        ).where(
            src == self.unk_id,
            1.0
        ).log()
        mask = padding_mask + unknown_mask

        return self.tokenizer.decode(self.sample(src, mask))

    @abstractmethod
    def sample(self, src: Tensor, mask: Tensor) -> list[int]:
        ...
