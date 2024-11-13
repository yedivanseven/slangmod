from enum import StrEnum
from typing import Literal

__all__ = [
    'Devices',
    'LiteralDevice',
    'Tokenizers',
    'Positions',
    'Scaling',
    'Styles',
    'Generators'
]

class Devices(StrEnum):
    CPU = 'cpu'
    CUDA = 'cuda'

type LiteralDevice = Literal['cpu', 'cuda']


class Tokenizers(StrEnum):
    BPE = 'bpe'
    WORDPIECE = 'wordpiece'
    UNIGRAM = 'unigram'


class Positions(StrEnum):
    SINUSOIDAL = 'sinusoidal'
    LEARNABLE = 'learnable'


class Scaling(StrEnum):
    INVERSE = 'inverse'
    COSINE = 'cosine'


class Styles(StrEnum):
    SPACE = 'space'
    PARAGRAPH = 'paragraph'
    QUOTE = 'quote'
    DIALOGUE = 'dialogue'


class Generators(StrEnum):
    GREEDY = 'greedy'
    RANDOM = 'random'
    TOP_K = 'top_k'
    TOP_P = 'top_p'
    BEAM = 'beam'
