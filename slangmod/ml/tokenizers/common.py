from tokenizers import Regex
from tokenizers.normalizers import Sequence
from tokenizers.normalizers import Strip, NFD, StripAccents, Replace
from ...config import config

__all__ = ['normalizer']

normalizer = Sequence([
    Strip(),
    NFD(),
    StripAccents(),
    Replace(Regex(config.tokens.eos_regex), config.tokens.eos_repl)
])
