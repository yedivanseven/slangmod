from tokenizers import AddedToken, Regex
from tokenizers.normalizers import Sequence
from tokenizers.normalizers import Strip, NFD, StripAccents, Replace
from ...config import config
from ...etl.regex import PARAGRAPH_REGEX

__all__ = [
    'PAD',
    'UNK',
    'EOS',
    'SPECIAL_TOKENS',
    'normalizer',
]

# Special tokens
PAD = AddedToken('[PAD]', single_word=True, special=True)
UNK = AddedToken('[UNK]', single_word=True, special=True)
EOS = AddedToken(
    config.tokens.eos_symbol,
    single_word=True,
    lstrip=True,
    rstrip=True,
    normalized=True,
    special=True
)
SPECIAL_TOKENS = [PAD, UNK, EOS]

# Normalizer
normalizer = Sequence([
    Strip(),
    NFD(),
    StripAccents(),
    Replace(Regex(PARAGRAPH_REGEX), config.tokens.eos_repl)
])
