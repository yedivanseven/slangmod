from enum import StrEnum

__all__ = [
    'Tokenizers',
    'Generators'
]

class Tokenizers(StrEnum):
    BPE = 'bpe'
    WORDPIECE = 'wordpiece'


class Generators(StrEnum):
    GREEDY = 'greedy'
    RANDOM = 'random'
    TOP_K = 'top_k'
    TOP_P = 'top_p'
    BEAM = 'beam'
