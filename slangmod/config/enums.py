from enum import StrEnum

__all__ = [
    'Tokenizers',
    'Positions',
    'Wrappers',
    'Generators'
]


class Tokenizers(StrEnum):
    BPE = 'bpe'
    WORDPIECE = 'wordpiece'


class Positions(StrEnum):
    SINUSOIDAL = 'sinusoidal'
    LEARNABLE = 'learnable'


class Wrappers(StrEnum):
    SIMPLE = 'simple'
    PARAGRAPH = 'paragraph'
    QUOTE = 'quote'
    DIALOGUE = 'dialogue'


class Generators(StrEnum):
    GREEDY = 'greedy'
    RANDOM = 'random'
    TOP_K = 'top_k'
    TOP_P = 'top_p'
    BEAM = 'beam'
