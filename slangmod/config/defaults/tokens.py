from swak.jsonobject import JsonObject
from swak.jsonobject.fields import Lower
from ..enums import Tokenizers


class Tokens(JsonObject):
    end_of_word_suffix = '</w>'
    pad_symbol = '[PAD]'
    unk_symbol = '[UNK]'
    eos_symbol = '[EOS]'

    algo: Lower() = Tokenizers.UNIGRAM
    vocab: int = 16384
    dropout: float = 0.0
    min_freq: int = 0
    max_len: int = 16
    shrink_factor: float = 0.75
    n_iter: int = 2

    eos_regex: str = '\n{2,}'
    eos_string: str = '\n\n'

    @property
    def eos_repl(self) -> str:
        """What should the `eos_string` be replaced with?"""
        return ' ' + self.eos_symbol.strip() + ' '
