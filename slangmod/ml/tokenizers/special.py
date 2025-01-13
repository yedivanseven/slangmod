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
EOS = AddedToken(config.tokens.eos_symbol, special=True, normalized=True)


class Special(ArgRepr):
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

    .. _tokenizers: https://huggingface.co/docs/tokenizers/index
    .. _AddedToken: https://huggingface.co/docs/tokenizers/api/added-tokens

    """

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
        """Special tokens in the correct order."""
        return [
            self.pad,
            self.unk,
            *self.unpredictable,
            self.eos,
            *self.predictable
        ]

    @property
    def id_to_token(self) -> dict[int, AddedToken]:
        """Dictionary from special token ids to the added tokens themselves."""
        return dict(enumerate(self.tokens))

    @property
    def pad_id(self) -> int:
        """Id of the padding token. Always 0."""
        return 0

    @property
    def unk_id(self) -> int:
        """Id of the unknown token. Always 1."""
        return 1

    @property
    def eos_id(self) -> int:
        """Id of the end-of-sequence token."""
        return self.tokens.index(self.eos)

    @property
    def unigram_vocab(self) -> list[tuple[str, float]]:
        """Initial vocabulary of special tokens for the Unigram tokenizer."""
        return [(token.content, 0.0) for token in self.tokens]

    def __iter__(self) -> Iterator[AddedToken]:
        return iter(self.tokens)

    def __len__(self) -> int:
        return len(self.tokens)


# Provide a ready-to-use instance of Special
special = Special(pad=PAD, unk=UNK, eos=EOS)
