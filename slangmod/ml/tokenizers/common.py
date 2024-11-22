from tokenizers import AddedToken, Regex
from tokenizers.normalizers import Sequence
from tokenizers.normalizers import Strip, NFKD, StripAccents, Replace
from ...config import config
from ...etl.regex import paragraph_regex

__all__ = [
    'PAD',
    'UNK',
    'EOS',
    'normalizer',
]

# Special tokens
PAD = AddedToken('[PAD]', single_word=True, special=True)
UNK = AddedToken('[UNK]', single_word=True, special=True)
EOS = AddedToken(
    config.tokens.eos_symbol,
    single_word=False,
    lstrip=True,
    rstrip=True,
    special=True
)

# Normalizer
normalizer = Sequence([
    Strip(),
    StripAccents(),
    NFKD(),
    Replace(Regex(paragraph_regex), config.tokens.eos_symbol)
])
