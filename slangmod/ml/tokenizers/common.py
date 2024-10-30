from tokenizers import AddedToken, Regex
from tokenizers.normalizers import Sequence
from tokenizers.normalizers import Strip, NFKC, StripAccents, Replace

__all__ = [
    'PAD',
    'UNK',
    'EOS',
    'normalizer',
]

# Special tokens
PAD = AddedToken('[PAD]', single_word=True, special=True)
UNK = AddedToken('[UNK]', single_word=True, special=True)
EOS = AddedToken('[EOS]', single_word=True, special=True, normalized=True)

# Normalizer
normalizer = Sequence([
    Strip(),
    NFKC(),
    StripAccents(),
    Replace(Regex(r'\n{2,}'), f' {EOS.content} '),
    Replace(r'“', r'"'),
    Replace(r'”', r'"'),
    Replace(r'‘', r"'"),
    Replace(r'’', r"'"),
])
