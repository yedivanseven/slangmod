from tokenizers import Regex
from tokenizers.normalizers import Sequence
from tokenizers.normalizers import Strip, NFD, StripAccents, Replace
from ...config import config
from ...etl.regex import PARAGRAPH_REGEX

__all__ = ['normalizer']

normalizer = Sequence([
    Strip(),
    NFD(),
    StripAccents(),
    Replace(Regex(PARAGRAPH_REGEX), config.tokens.eos_repl)
])
