"""Helpers and wrappers to safely use HuggingFace `tokenizers <https://
huggingface.co/docs/tokenizers/index>`_."""

from .special import Special, special
from .algo import Algo
from .bpe import bpe
from .wordpiece import wordpiece
from .unigram import unigram
from ...config import config, Tokenizers

__all__ = [
    'Special',
    'special',
    'Algo',
    'tokenizer'
]

tokenizer = {
    Tokenizers.BPE: bpe,
    Tokenizers.WORDPIECE: wordpiece,
    Tokenizers.UNIGRAM: unigram
}[config.tokens.algo]
