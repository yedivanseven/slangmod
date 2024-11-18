from typing import Self, Any, overload
from tokenizers import Tokenizer
from tokenizers.models import Model
from tokenizers.normalizers import Normalizer
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers.processors import PostProcessor
from tokenizers.decoders import Decoder
from tokenizers.tokenizers import AddedToken
from tokenizers.trainers import Trainer
from .common import PAD, UNK, EOS


__all__ = ['Algo']


class Algo:

    def __init__(
            self,
            model: Model,
            trainer: Trainer,
            normalizer: Normalizer | None = None,
            pre_tokenizer: PreTokenizer | None = None,
            post_processor: PostProcessor | None = None,
            decoder: Decoder | None = None,
            pad: AddedToken = PAD,
            unk: AddedToken = UNK,
            eos: AddedToken = EOS,
            *extra: AddedToken,
    ) -> None:
        self.trainer = trainer
        self.pad = pad
        self.unk = unk
        self.eos = eos
        self.extra = extra
        self.tokenizer = Tokenizer(model)
        self.tokenizer.normalizer = normalizer
        self.tokenizer.pre_tokenizer = pre_tokenizer
        self.tokenizer.post_processor = post_processor
        self.tokenizer.decoder = decoder
        self.tokenizer.add_tokens(self.special)
        self.tokenizer.add_special_tokens(self.special)

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
        tokenizer = Tokenizer.from_file(path)
        added_tokens = tokenizer.get_added_tokens_decoder()
        sorted_tokens = sorted(added_tokens.items(), key=lambda x: x[0])
        pad, unk, *extra, eos = (token for _, token in sorted_tokens)
        algo = self.__class__(
            model=tokenizer.model,
            trainer=self.trainer,
            normalizer=tokenizer.normalizer,
            pre_tokenizer=tokenizer.pre_tokenizer,
            post_processor=tokenizer.post_processor,
            decoder=tokenizer.decoder,
            pad=pad,
            unk=unk,
            eos=eos,
            *extra
        )
        algo.tokenizer = tokenizer
        return algo

    def train(self, files: list[str], trainer: Trainer | None = None) -> Self:
        trainer_to_use = self.trainer if trainer is None else trainer
        self.tokenizer.train(files, trainer_to_use)
        return self

    def __call__(
            self,
            sequence: str | list[str],
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
            encodings = self.tokenizer.encode_batch(
                sequence,
                is_pretokenized,
                add_special_tokens
            )
        return [encoding.ids for encoding in encodings]
