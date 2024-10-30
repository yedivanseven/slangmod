from swak.jsonobject import JsonObject
from swak.jsonobject.fields import Lower


class Tokens(JsonObject):
    algo: Lower() = 'bpe'   # 'bpe', 'wordpiece', or 'unigram'
    vocab: int = 2048  # 4096  # 8192
    dropout: float = 0.0
    min_frequency: int = 0
    max_length: int = 16
    shrink_factor: float = 0.75
    n_iter: int = 2
