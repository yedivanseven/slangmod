import re
from typing import Self, Any
from collections.abc import Iterable
from tokenizers.normalizers import Normalizer
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers.models import Model, Unigram
from tokenizers.processors import PostProcessor
from tokenizers.decoders import Decoder
from tokenizers.trainers import Trainer, UnigramTrainer
from tokenizers import Tokenizer
from .special import Special

__all__ = ['Algo']

type Documents = Iterable[str | list[str]]
type Sequences = Iterable[ str | list[str] | tuple[str | list[str], str]]


class Algo:
    """Wrap a Tokenizer instance for safe, convenient, and consistent usage.

    In particular, the ``train`` and ``train_from_iterator`` methods are
    overwritten to always use the `trainer` provided at instantiation of
    this wrapper (and to return the trained instance). The ``from_*`` methods
    are all overwritten to return an instance of this wrapper instead of a
    bare tokenizer. All other method calls are simply forwarded to the
    wrapped ``Tokenizer``, so that instances serve as a drop-in replacement.

    Parameters
    ----------
    special: Special
        An instance of ``Special``, specifying all the special tokens that
        the tokenizer, the model, and the trainer should be aware of.
    tokenizer: Model or Tokenizer
        Instance of a `Model`_ or a `Tokenizer`_ from the HuggingFace
        `tokenizers`_ package. If it is a ``Model``, a fresh ``Tokenizer``
        instance will be created from it.
    trainer: Trainer, optional
        An instance of a `Trainer`_ from the HuggingFace `tokenizers`_
        package. If not given, an appropriate trainer will be instantiated
        with its default parameters. Defaults to ``None``
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

    Important
    ---------
    While the consistency of the special tokens used by the `tokenizer` and
    the `trainer` will be checked for. No such guarantees can be made
    regarding their potential use in the `normalizer`, the `decoder`,
    or the `post_processor`.

    .. _tokenizers: https://huggingface.co/docs/tokenizers/index
    .. _Model: https://huggingface.co/docs/tokenizers/api/models
    .. _Tokenizer: https://huggingface.co/docs/tokenizers/api/tokenizer
    .. _Trainer: https://huggingface.co/docs/tokenizers/api/trainers
    .. _Normalizer: https://huggingface.co/docs/tokenizers/api/normalizers
    .. _PreTokenizer: https://huggingface.co/docs/tokenizers/api/pre-tokenizers
    .. _Decoder: https://huggingface.co/docs/tokenizers/api/decoders
    .. _Processor: https://huggingface.co/docs/tokenizers/api/post-processors

    """

    _UNK_ID_REGEX = re.compile(r'unk_id=([0-9]+?),')
    _UNK_TOKEN_REGEX = re.compile(r'unk_token=(.+?),')

    def __init__(
            self,
            special: Special,
            tokenizer: Model | Tokenizer,
            trainer: Trainer | None = None,
            normalizer: Normalizer | None = None,
            pre_tokenizer: PreTokenizer | None = None,
            decoder: Decoder | None = None,
            post_processor: PostProcessor | None = None
    ) -> None:
        self.special = special
        self.tokenizer = self.__valid(tokenizer)
        self.trainer = self.__compatible(trainer)
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

    def __getattr__(self, attr: str) -> Any:
        """Redirect attribute/method access to the wrapped tokenizer."""
        # This is needed for un-pickling to work
        if attr == '__setstate__':
            cls = self.__class__.__name__
            msg = f"'{cls}' object has no attribute {attr}"
            raise AttributeError(msg)
        # This is the actual forwarding
        return getattr(self.tokenizer, attr)

    @property
    def vocab(self) -> int:
        """The target vocabulary size of the wrapped trainer."""
        return self.trainer.vocab_size

    @property
    def unk_id(self) -> int:
        """The ID of the unknown token. Always 1."""
        return self.special.unk_id

    @property
    def eos_id(self) -> int:
        """The ID of the end-of-sequence token."""
        return self.special.eos_id

    def from_buffer(self, buffer: bytes) -> Self:
        """Instantiate a new algo from the given buffer.

        Parameters
        ----------
        buffer: bytes
            A buffer containing a previously serialized ``Tokenizer``.

        Returns
        -------
        Algo
            The new algo.

        """
        return self.__class__(
            self.special,
            Tokenizer.from_buffer(buffer),
            self.trainer
        )

    def from_file(self, path: str) -> Self:
        """Instantiate a new Tokenizer from the file at the given path.

        Parameters
        ----------
        path: str
            Full path to the tokenizer file to load.

        Returns
        -------
        Algo
            The new algo.

        """
        return self.__class__(
            self.special,
            Tokenizer.from_file(path),
            self.trainer
        )

    def from_pretrained(
            self,
            identifier: str,
            revision: str = 'main',
            token: str | None = None
    ) -> Self:
        """Instantiate a new Tokenizer from pulled from the HuggingFace Hub.

        Parameters
        ----------
        identifier: str
            The identifier of a Model on the HuggingFace Hub that contains a
            "tokenizer.json" file.
        revision: str, optional
            A branch or commit id. Defaults to "main".
        token: str, optional
            An optional auth token used to access private repositories
            on the HuggingFace Hub.

        Returns
        -------
        Algo
            The new algo.

        Warnings
        --------
        Because this package does not depend on the HuggingFace `transformers
        <https://huggingface.co/docs/transformers/index>`_ package, this
        method not might work as anticipated or simply not work at all.

        """
        return self.__class__(
            self.special,
            Tokenizer.from_pretrained(identifier, revision, token),
            self.trainer
        )

    def from_str(self, json: str) -> Self:
        """Instantiate a new Tokenizer from the given JSON string.

        Parameters
        ----------
        json: str
            A valid JSON string representing a previously serialized Tokenizer.

        Returns
        -------
        Algo
            The new algo.

        """
        return self.__class__(
            self.special,
            Tokenizer.from_str(json),
            self.trainer
        )

    def train(self, files: list[str]) -> Self:
        """Train the wrapped tokenizer on the given files.

        Parameters
        ----------
        files: list of str
            A list of paths to the files that should be used for training.

        Returns
        -------
        Algo
            Itself with the wrapped ``Tokenizer`` now trained.

        """
        self.tokenizer.train(files, self.trainer)
        return self

    def train_from_iterator(self, documents: Documents) -> Self:
        """Train the wrapped tokenizer on the provided iterator over documents.

        Parameters
        ----------
        documents: iterable over str or over list of str
            The documents to train on.

        Returns
        -------
        Algo
            Itself with the wrapped ``Tokenizer`` now trained.

        """
        self.tokenizer.train_from_iterator(documents, self.trainer)
        return self

    def terminate(self, encodings: list[int]) -> list[int]:
        """Add the id of the end-of-sequence token to a list of integers.

        If the last integer in the list already is the id of the end-of-
        sequence token, the original list is returned.

        Parameters
        ----------
        encodings: list of int
            An encoded piece of text.

        Returns
        -------
        list of int
            The input `encodings`, potentially extended by the id of the
            end-of-sequence token.

        """
        if encodings and encodings[-1] == self.eos_id:
            return encodings
        encodings.append(self.eos_id)
        return encodings

    def __call__(self, sequences: Sequences) -> list[list[int]]:
        """Wrap combined call to encode a batch sequences and terminate them.

        The arguments `is_pretokenized` and `add_special_tokens` are kept
        at ``False`` and ``True``, respectively.

        Parameters
        ----------
        sequences: iterable
            Batch of documents to encode. See the `documentation <https://
            huggingface.co/docs/tokenizers/api/tokenizer#tokenizers.Tokenizer
            .encode_batch_fast`__ for possible formats.

        Returns
        -------
        list of list of int
            The encoded documents, all terminated with the id of the
            end-of-sequence token.

        """
        return [
            self.terminate(encoding.ids)
            for encoding in self.tokenizer.encode_batch_fast(
                sequences,
                is_pretokenized=False,
                add_special_tokens=True
            )
        ]

    def __valid(self, tokenizer: Tokenizer | Model) -> Tokenizer:
        """Check the special tokens in tokenizer and model for consistency."""
        if isinstance(tokenizer, Model):
            tokenizer = Tokenizer(tokenizer)

        # First, we add the desired special tokens (nothing
        # should happen if we add the same tokens again).
        tokenizer.add_special_tokens(self.special.tokens)
        # Then, we get ALL added tokens ...
        added = tokenizer.get_added_tokens_decoder()
        # ... filter for special ones ...
        specials = filter(lambda item: item[1].special, added.items())
        # ... get at most as many as we need, ...
        initials = sorted(specials)[:len(self.special)]
        # ... and check if they are consistent with what we specified.
        special_tokens_are_inconsistent = initials != self.special.items
        # Finally, we raise if they are not.
        if special_tokens_are_inconsistent:
            cls = tokenizer.model.__class__.__name__
            actual = {i: token.content for i, token in initials}
            tmp = "Special tokens {} of the {} tokenizer don't match {}!"
            msg = tmp.format(actual, cls, self.special)
            raise ValueError(msg)

        # Also the models themselves make use of the token for "unknown".
        # The Unigram model is special in that sense.
        if isinstance(tokenizer.model, Unigram):
            actual = self.__unk_id(tokenizer.model)
            expected = self.special.unk_id
        # For all others, we check if it is set, and set it if it is not.
        else:
            expected = self.special.unk.content
            if tokenizer.model.unk_token is None:
                tokenizer.model.unk_token = expected
            actual = tokenizer.model.unk_token
        # If the settings are still inconsistent, we raise.
        if actual != expected:
            cls = tokenizer.model.__class__.__name__
            tmp = 'Unknown token "{}" in {} model must be set to "{}"!'
            msg = tmp.format(actual, cls, expected)
            raise ValueError(msg)

        return tokenizer

    def __unk_id(self, model: Unigram) -> int:
        """Extract unk_id from the Unigram model string representation."""
        return int(self._UNK_ID_REGEX.search(str(model)).group(1))

    def __unk_token(self, trainer: UnigramTrainer) -> str:
        """Extract unk_token from the UnigramTrainer string representation."""
        return self._UNK_TOKEN_REGEX.search(str(trainer)).group(1).strip('"')

    def __compatible(self, trainer: Trainer | None) -> Trainer:
        """Check the provided trainer for consistency with the tokenizer."""
        # Create a new trainer with default settings and, ...
        model_trainer = UnigramTrainer(
            unk_token=self.special.unk.content
        ) if isinstance(
            self.tokenizer.model, Unigram
        ) else self.tokenizer.model.get_trainer()
        # ... if no trainer was provided, return it.
        if trainer is None:
            model_trainer.special_tokens = self.special.tokens
            return model_trainer

        # Check that the algorithms of tokenizer and trainer match.
        if not isinstance(trainer, model_trainer.__class__):
            model_cls = self.tokenizer.model.__class__.__name__
            trainer_cls = trainer.__class__.__name__
            tmp = '{} model and {} are incompatible!'
            msg = tmp.format(model_cls, trainer_cls)
            raise TypeError(msg)

        # The UnigramTrainer has an attribute unk_token
        if isinstance(trainer, UnigramTrainer):
            actual = self.__unk_token(trainer)
            expected = self.special.unk.content
            if actual != expected:
                tmp = 'Unknown token "{}" in UnigramTrainer should be "{}"!'
                msg = tmp.format(actual, expected)
                raise ValueError(msg)

        # Set the required special tokens if the trainer has none.
        if trainer.special_tokens is None:
            trainer.special_tokens = self.special.tokens
            return trainer

        # Check if the special tokens in the trainer are consistent ...
        inconsistent = trainer.special_tokens != self.special.tokens
        # ... and raise if they are not.
        if inconsistent:
            contents = {
                i: token.content if hasattr(token, 'content') else token
                for i, token in enumerate(trainer.special_tokens)
            }
            trainer_cls = trainer.__class__.__name__
            tmp = "Special tokens {} of the {} don't match {}!"
            msg = tmp.format(contents, trainer_cls, self.special)
            raise ValueError(msg)

        return trainer
