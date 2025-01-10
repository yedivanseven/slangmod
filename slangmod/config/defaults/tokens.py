from swak.jsonobject import JsonObject
from swak.jsonobject.fields import Lower
from ..enums import Tokenizers


class Tokens(JsonObject):
    encoding: Lower() = 'cp1252'
    algo: Lower() = Tokenizers.UNIGRAM
    vocab: int = 16384
    dropout: float = 0.0
    min_frequency: int = 0
    max_length: int = 16
    shrink_factor: float = 0.75
    n_iter: int = 2
    pad_symbol = '[PAD]'
    pad_id = 0
    unk_symbol = '[UNK]'
    unk_id = 1
    eos_symbol: str = '[EOS]'
    eos_string: str = '\n\n'

    @property
    def eos_repl(self) -> str:
        return ' ' + self.eos_symbol.strip() + ' '
