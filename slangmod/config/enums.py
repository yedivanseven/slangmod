from enum import StrEnum
from typing import Literal

__all__ = [
    'Devices',
    'LiteralDevice',
    'Tokenizers',
    'Positions',
    'Optimizers',
    'Scaling',
    'Models',
    'LiteralModel',
    'Generators',
    'Styles'
]

type LiteralDevice = Literal['cpu', 'cuda']
type LiteralModel = Literal['vanilla']


class Devices(StrEnum):
    CPU = 'cpu'
    CUDA = 'cuda'


class Tokenizers(StrEnum):
    BPE = 'bpe'
    WORDPIECE = 'wordpiece'
    UNIGRAM = 'unigram'


class Positions(StrEnum):
    SINUSOIDAL = 'sinusoidal'
    ROTARY = 'rotary'
    LEARNABLE = 'learnable'
    NONE = 'none'


class Optimizers(StrEnum):
    ADMAW = 'adamw'
    ADAFACTOR = 'adafactor'


class Scaling(StrEnum):
    INVERSE = 'inverse'
    EXPONENTIAL = 'exponential'
    COSINE = 'cosine'


class Models(StrEnum):
    VANILLA = 'vanilla'


class Generators(StrEnum):
    GREEDY = 'greedy'
    RANDOM = 'random'
    TOP_K = 'top_k'
    TOP_P = 'top_p'
    BEAM = 'beam'


class Styles(StrEnum):
    SPACE = 'space'
    PARAGRAPH = 'paragraph'
    QUOTE = 'quote'
    DIALOGUE = 'dialogue'
