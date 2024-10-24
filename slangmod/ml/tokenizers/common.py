from tokenizers import AddedToken
from tokenizers.normalizers import Sequence as NormalizerSequence
from tokenizers.normalizers import Strip, NFD, StripAccents, Replace
from tokenizers.pre_tokenizers import Sequence as PreTokenizerSequence
from tokenizers.pre_tokenizers import WhitespaceSplit, Digits

__all__ = [
    'special_tokens',
    'normalizer',
    'pre_tokenizer'
]

# Special tokens
pad = AddedToken('[PAD]', single_word=True, special=True)
unk = AddedToken('[UNK]', single_word=True, special=True)
eos = AddedToken('[EOS]', single_word=True, special=True, normalized=True)
special_tokens = [pad, unk, eos]

# Pipeline steps
normalizer = NormalizerSequence([
    Strip(),
    NFD(),
    StripAccents(),
    Replace('\n\n', ' [EOS] '),
])
pre_tokenizer = PreTokenizerSequence([
    WhitespaceSplit(),
    Digits(),
])
