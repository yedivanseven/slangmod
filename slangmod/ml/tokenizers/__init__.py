from .algo import Algo
from .bpe import bpe
from .wordpiece import wordpiece
from .unigram import unigram
from ... config import config, Tokenizers

__all__ = [
    'Algo',
    'bpe',
    'wordpiece',
    'unigram',
    'tokenizer'
]

tokenizer = {
    Tokenizers.BPE: bpe,
    Tokenizers.WORDPIECE: wordpiece,
    Tokenizers.UNIGRAM: unigram
}[config.tokens.algo]

# ToDo: Investigate whitespaces in number encoding and decoding!
# ToDo: Investigate why EOS is in the vocabulary!
