from collections.abc import Iterable, Iterator
from typing import overload
from tokenizers import AddedToken
from ...config import config

__all__ = [
    'Special',
    'special'
]

PAD = AddedToken(config.tokens.pad_symbol, special=True)
UNK = AddedToken(config.tokens.unk_symbol, special=True)
EOS = AddedToken(config.tokens.eos_symbol, special=True, normalized=True)


class Special:
    """Wrapper around special tokens for consistent access across projects.

    Parameters
    ----------
    unpredictable: Iterable of AddedToken
        All special tokens, in the form of `AddedToken`_ from the HuggingFace
        `tokenizers`_ package, that a language model might get as input,
        but will never be required to predict (e.g., mask or unknown).
        Defaults to an empty tuple. If given, these tokens will come after
        `pad` and `unk` in the vocabulary (i.e., their indices will be
        2 and up), but before `eos`.
    *predictable: AddedToken
        Additional special tokens, in the form of `AddedToken`_ from the
        HuggingFace `tokenizers`_ package, that the model might be
        required to predict. If given, this tokens will come after `eos`.
    pad: AddedToken
        The padding token, in the form of an `AddedToken`_ from the
        HuggingFace `tokenizers`_ package. It will always have token
        index 0.
    unk: AddedToken
        The unknown token, in the form of an `AddedToken`_ from the
        HuggingFace `tokenizers`_ package. It will always have token
        index 1.
    eos: AddedToken
        The end-of-sequence token, in the form of an `AddedToken`_ from the
        HuggingFace `tokenizers`_ package. This token comes after the
        `unpredictable` tokens, but before the `predictable` tokens.

    Important
    ---------
    Various token IDs and string representations are needed at seemingly
    disjoint parts of the overall workflow. It is, therefore, deceptively easy
    to make a mistake somewhere, somehow. Instances of this class are to serve
    as the single ground truth for your entire project.

    .. _AddedToken: https://huggingface.co/docs/tokenizers/api/added-tokens
    .. _tokenizers: https://huggingface.co/docs/tokenizers/index

    """

    def __init__(
            self,
            unpredictable: Iterable[AddedToken] = (),
            *predictable: AddedToken,
            pad: AddedToken,
            unk: AddedToken,
            eos: AddedToken,
    ) -> None:
        self.__validate(*unpredictable, *predictable, pad, unk, eos)
        self.unpredictable = tuple(unpredictable)
        self.predictable = predictable
        self.pad = pad
        self.unk = unk
        self.eos = eos

    @staticmethod
    def __validate(*tokens: AddedToken) -> None:
        """Ensure that all provided tokens are, in fact, special tokens."""
        invalid = [token.content for token in tokens if not token.special]
        if len(invalid) == 1:
            raise TypeError(f'Token {invalid[0]} is not special!')
        if len(invalid) > 1:
            raise TypeError(f'Tokens {', '.join(invalid)} are not special!')

    def __str__(self) -> str:
        return str(dict(zip(self.ids, self.contents)))

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        contents = ', '.join([token.content for token in self])
        return f'{cls}({contents})'

    def __iter__(self) -> Iterator[AddedToken]:
        return iter(self.tokens)

    def __len__(self) -> int:
        return len(self.tokens)

    @overload
    def __getitem__(self, token_id: int) -> AddedToken:
        ...

    @overload
    def __getitem__(self, token_id: slice) -> list[AddedToken]:
        ...

    def __getitem__(self, token_id):
        return self.tokens[token_id]

    @property
    def tokens(self) -> list[AddedToken]:
        """Special tokens in the correct order."""
        return [
            self.pad,
            self.unk,
            *self.unpredictable,
            self.eos,
            *self.predictable
        ]

    @property
    def ids(self) -> list[int]:
        """Only the IDs of all special tokens."""
        return list(range(len(self.tokens)))

    @property
    def contents(self) -> list[str]:
        """Only the string representations of all special tokens."""
        return [token.content for token in self.tokens]

    @property
    def items(self) -> list[tuple[int, AddedToken]]:
        """Tuples of (ID, added token) for all special tokens"""
        return list(zip(self.ids, self.tokens))

    @property
    def decoder(self) -> dict[int, AddedToken]:
        """Dictionary with indices as keys and special tokens as values."""
        return dict(zip(self.ids, self.tokens))

    @property
    def encoder(self) -> dict[str, int]:
        """Special-token strings as keys and their indices as values."""
        return dict(zip(self.contents, self.ids))

    @property
    def pad_id(self) -> int:
        """ID of the padding token. Always 0."""
        return 0

    @property
    def unk_id(self) -> int:
        """ID of the unknown token. Always 1."""
        return 1

    @property
    def eos_id(self) -> int:
        """ID of the end-of-sequence token."""
        return self.tokens.index(self.eos)

    @property
    def unigram_vocab(self) -> list[tuple[str, float]]:
        """Initial vocabulary of special tokens for the Unigram tokenizer."""
        return [(token.content, 0.0) for token in self.tokens]


# Provide a ready-to-use instance of Special
special = Special(pad=PAD, unk=UNK, eos=EOS)
