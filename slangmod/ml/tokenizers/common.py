from tokenizers import AddedToken
from tokenizers.normalizers import Sequence
from tokenizers.normalizers import Strip, NFD, StripAccents, Replace

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
    NFD(),
    StripAccents(),
    Replace('\n\n', f' {EOS.content} '),
])
