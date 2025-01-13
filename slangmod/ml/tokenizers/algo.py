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
    """Wrap a Tokenizer instance for save, convenient, and consistent usage.

    In particular, the ``train`` and ``train_from_iterator`` methods are
    overwritten to always use the `trainer` provided at instantiation of
    this wrapper, and the ``from_*`` methods are overwritten to return an
    instance of this wrapper instead of a tokenizer. All other method calls
    are simply be forwarded to the underlying ``Tokenizer``.

    Parameters
    ----------
    model: Model or Tokenizer
        Instance of a `Model`_ or a `Tokenizer`_ from the HuggingFace
        `tokenizers`_ package. If it is a ``Model``, a fresh ``Tokenizer``
        instance will be created from it.
    trainer: Trainer
        An instance of a `Trainer`_ from the HuggingFace `tokenizers`_
        package.
    unk_id: int
        The token index that should be used to encode unknown symbols.
    eos_id: int
        The token index to use for indicating the end of a sequence.
    normalizer: Normalizer, optional
        An instance of a `Normalizer`_ from the HuggingFace `tokenizers`_
        package. Defaults to ``None``.
    pre_tokenizer: PreTokenizer, optional
        An instance of a `PreTokenizer`_ from the HuggingFace `tokenizers`_
        package. Defaults to ``None``.
    decoder: Decoder, optional
        An instance of a `Decoder`_ from the HuggingFace `tokenizers`_
        package. Defaults to ``None``.
    post_processor: Processor, optional
        An instance of a `Processor`_ from the HuggingFace `tokenizers`_
        package. Defaults to ``None``.

    .. _tokenizers: https://huggingface.co/docs/tokenizers/index
    .. _Model: https://huggingface.co/docs/tokenizers/api/models
    .. _Tokenizer: https://huggingface.co/docs/tokenizers/api/tokenizer
    .. _Trainer: https://huggingface.co/docs/tokenizers/api/trainers
    .. _Normalizer: https://huggingface.co/docs/tokenizers/api/normalizers
    .. _PreTokenizer: https://huggingface.co/docs/tokenizers/api/pre-tokenizers
    .. _Decoder: https://huggingface.co/docs/tokenizers/api/decoders
    .. _Processor: https://huggingface.co/docs/tokenizers/api/post-processors

    """

    def __init__(
            self,
            model: Model | Tokenizer,
            trainer: Trainer,
            unk_id: int,
            eos_id: int,
            normalizer: Normalizer | None = None,
            pre_tokenizer: PreTokenizer | None = None,
            decoder: Decoder | None = None,
            post_processor: PostProcessor | None = None
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
        if decoder is not None:
            self.tokenizer.decoder = decoder
        if post_processor is not None:
            self.tokenizer.post_processor = post_processor

    def __repr__(self) -> str:
        cls = self.tokenizer.model.__class__.__name__
        return f'{cls}(...)'

    @property
    def vocab(self) -> int:
        """The vocabulary size of the wrapped tokenizer."""
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

    def from_buffer(self, buffer: bytes) -> Self:
        return self.__class__(
            Tokenizer.from_buffer(buffer),
            self.trainer,
            self.unk_id,
            self.eos_id
        )

    def from_file(self, path: str) -> Self:
        return self.__class__(
            Tokenizer.from_file(path),
            self.trainer,
            self.unk_id,
            self.eos_id
        )

    def from_pretrained(
            self,
            identifier: str,
            revision: str = 'main',
            token: str | None = None
    ) -> Self:
        return self.__class__(
            Tokenizer.from_pretrained(identifier, revision, token),
            self.trainer,
            self.unk_id,
            self.eos_id
        )

    def from_str(self, json: str) -> Self:
        return self.__class__(
            Tokenizer.from_str(json),
            self.trainer,
            self.unk_id,
            self.eos_id
        )

    def train(self, files: list[str]) -> Self:
        self.tokenizer.train(files, self.trainer)
        return self

    def train_from_iterator(self, docs: Iterable[str]) -> Self:
        self.tokenizer.train_from_iterator(docs, self.trainer)
        return self

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
