from collections.abc import Iterable, Iterator
from tokenizers import AddedToken
from swak.misc import ArgRepr
from ...config import config

__all__ = [
    'Special',
    'special'
]

PAD = AddedToken(config.tokens.pad_symbol, special=True)
UNK = AddedToken(config.tokens.unk_symbol, special=True)
EOS = AddedToken(config.tokens.eos_symbol, normalized=True, special=True)


class Special(ArgRepr):

    def __init__(
            self,
            unpredictable: Iterable[AddedToken] = (),
            *predictable: AddedToken,
            pad: AddedToken,
            unk: AddedToken,
            eos: AddedToken,
    ) -> None:
        self.unpredictable = tuple(unpredictable)
        self.predictable = predictable
        self.pad = pad
        self.unk = unk
        self.eos = eos
        super().__init__(
            tuple(token.content for token in self.unpredictable),
            tuple(token.content for token in self.predictable),
            pad=self.pad.content,
            unk=self.unk.content,
            eos=self.eos.content,
        )

    @property
    def tokens(self) -> list[AddedToken]:
        return [
            self.pad,
            self.unk,
            *self.unpredictable,
            self.eos,
            *self.predictable
        ]

    @property
    def pad_id(self) -> int:
        return 0

    @property
    def unk_id(self) -> int:
        return 1

    @property
    def eos_id(self) -> int:
        return self.tokens.index(self.eos)

    @property
    def unigram_vocab(self) -> list[tuple[str, float]]:
        return [(token.content, 0.0) for token in self.tokens]

    def __iter__(self) -> Iterator[AddedToken]:
        return iter(self.tokens)

    def __len__(self) -> int:
        return len(self.tokens)


# Provide a ready-to-use instance of Special
special = Special(pad=PAD, unk=UNK, eos=EOS)
