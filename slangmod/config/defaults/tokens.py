from swak.jsonobject import JsonObject
from swak.jsonobject.fields import Lower
from ..enums import Tokenizers


class Tokens(JsonObject):
    encoding: Lower() = 'cp1252'
    algo: Lower() = Tokenizers.BPE
    vocab: int = 32768
    dropout: float = 0.0
    min_frequency: int = 0
    max_length: int = 16
    shrink_factor: float = 0.75
    n_iter: int = 2
    eos_symbol: str = '[EOS]'
    eos_regex: str = r'\n{2,}'
    eos_string: str = '\n\n'
