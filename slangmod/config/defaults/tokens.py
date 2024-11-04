from swak.jsonobject import JsonObject
from swak.jsonobject.fields import Lower
from ..enums import Tokenizers


class Tokens(JsonObject):
    algo: Lower() = Tokenizers.BPE
    vocab: int = 2048
    dropout: float = 0.0
    min_frequency: int = 0
    max_length: int = 16
    shrink_factor: float = 0.75
    n_iter: int = 2
