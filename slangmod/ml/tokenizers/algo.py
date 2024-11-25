from typing import Self, Any
from collections.abc import Iterable
from tokenizers import Tokenizer
from tokenizers.normalizers import Normalizer
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers.processors import PostProcessor
from tokenizers.decoders import Decoder
from tokenizers.tokenizers import AddedToken
from tokenizers.trainers import Trainer


__all__ = ['Algo']


class Algo:

    def __init__(
            self,
            tokenizer: Tokenizer,
            trainer: Trainer,
            normalizer: Normalizer | None = None,
            pre_tokenizer: PreTokenizer | None = None,
            post_processor: PostProcessor | None = None,
            decoder: Decoder | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.trainer = trainer
        self.pad, self.unk, *self.extra, self.eos = trainer.special_tokens
        if normalizer is not None:
            self.tokenizer.normalizer = normalizer
        if pre_tokenizer is not None:
            self.tokenizer.pre_tokenizer = pre_tokenizer
        if post_processor is not None:
            self.tokenizer.post_processor = post_processor
        if decoder is not None:
            self.tokenizer.decoder = decoder

    def __repr__(self) -> str:
        cls = self.tokenizer.model.__class__.__name__
        return f'{cls}(...)'

    def __getattr__(self, attr: str) -> Any:
        return getattr(self.tokenizer, attr)

    @property
    def special(self) -> list[AddedToken]:
        return [self.pad, self.unk, *self.extra, self.eos]

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def unk_id(self) -> int:
        return 1

    @property
    def eos_id(self) -> int:
        return len(self.special) - 1

    @property
    def vocab(self) -> int:
        return self.tokenizer.get_vocab_size()

    def from_file(self, path: str) -> Self:
        return self.__class__(Tokenizer.from_file(path), self.trainer)

    def train(self, docs: Iterable[str]) -> Self:
        self.tokenizer.train_from_iterator(docs, self.trainer)
        return self

    def __call__(
            self,
            sequence: str | Iterable[str],
            pair: str | None = None,
            is_pretokenized: bool = False,
            add_special_tokens: bool = True
    ) -> list[list[int]]:
        if isinstance(sequence, str):
            encodings = [self.tokenizer.encode(
                sequence,
                pair,
                is_pretokenized,
                add_special_tokens
            )]
        else:
            encodings = self.tokenizer.encode_batch_fast(
                list(sequence),
                is_pretokenized,
                add_special_tokens
            )
        return [encoding.ids + [self.eos_id] for encoding in encodings]
