from swak.jsonobject import JsonObject
from swak.jsonobject.fields import Lower


class Tokenizer(JsonObject):
    algo: Lower() = 'bpe'   # 'bpe' or 'wordpiece'
    vocab_size: int = 4096  # 8192
    dropout: float = 0.0
