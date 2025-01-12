from typing import Self, Any
from collections.abc import Iterable
from tokenizers.normalizers import Normalizer
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers.models import Model
from tokenizers.processors import PostProcessor
from tokenizers.decoders import Decoder
from tokenizers.trainers import Trainer
from tokenizers import Tokenizer

__all__ = ['Algo']


class Algo:

    def __init__(
            self,
            model: Model | Tokenizer,
            trainer: Trainer,
            unk_id: int,
            eos_id: int,
            normalizer: Normalizer | None = None,
            pre_tokenizer: PreTokenizer | None = None,
            post_processor: PostProcessor | None = None,
            decoder: Decoder | None = None,
    ) -> None:
        if isinstance(model, Tokenizer):
            self.tokenizer = model
        else:
            self.tokenizer = Tokenizer(model)
        self.trainer = trainer
        self.eos_id = eos_id
        self.unk_id = unk_id
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

    @property
    def vocab(self) -> int:
        return self.tokenizer.get_vocab_size()

    def __getattr__(self, attr: str) -> Any:
        """Redirect attribute/method access to the wrapped tokenizer."""
        # This is needed for un-pickling to work
        if attr == '__setstate__':
            cls = self.__class__.__name__
            msg = f"'{cls}' object has no attribute {attr}"
            raise AttributeError(msg)
        # This is the actual forwarding
        return getattr(self.tokenizer, attr)

    def from_file(self, path: str) -> Self:
        return self.__class__(
            Tokenizer.from_file(path),
            self.trainer,
            self.unk_id,
            self.eos_id
        )

    def train(self, docs: Iterable[str]) -> Self:
        self.tokenizer.train_from_iterator(docs, self.trainer)
        return self

    def save(self, path: str, pretty: bool = True) -> None:
        self.tokenizer.save(path, pretty)

    def terminate(self, encodings: list[int]) -> list[int]:
        if encodings and encodings[-1] == self.eos_id:
            return encodings
        return encodings + [self.eos_id]

    def __call__(
            self,
            sequence: str | Iterable[str],
            pair: str | None = None,
            is_pretokenized: bool = False
    ) -> list[list[int]]:
        if isinstance(sequence, str):
            encodings = [self.tokenizer.encode(
                sequence,
                pair,
                is_pretokenized,
                add_special_tokens=True
            )]
        else:
            encodings = self.tokenizer.encode_batch_fast(
                list(sequence),
                is_pretokenized,
                add_special_tokens=True
            )
        return [self.terminate(encoding.ids) for encoding in encodings]
