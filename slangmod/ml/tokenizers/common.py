from tokenizers import AddedToken, Regex
from tokenizers.normalizers import Sequence
from tokenizers.normalizers import Strip, NFKC, StripAccents, Replace
from ...config import config

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
    single_word=True,
    special=True,
    normalized=True  # Ensures that explicit "[EOS]" gets tokenized as such
)

# Normalizer
normalizer = Sequence([
    Strip(),
    NFKC(),
    StripAccents(),
    Replace(Regex(config.tokens.eos_regex), f' {EOS.content} '),
    Replace(r'“', r'"'),
    Replace(r'”', r'"'),
    Replace(r'‘', r"'"),
    Replace(r'’', r"'"),
])
