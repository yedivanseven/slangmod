from .encoding import EncodingEnforcer, enforce_encoding
from .splitter import CorpusSplitter, split_corpus
from .folder import SequenceFolder, fold_train, fold_test
from .cleaner import CorpusCleaner
from .regex import (
    RegexReplacer,
    replace_article,
    replace_section,
    replace_newline,
    replace_minutes,
    replace_seconds,
    PARAGRAPH_REGEX,
    replace_single_quote,
    replace_double_quote
)

__all__ = [
    'EncodingEnforcer',
    'enforce_encoding',
    'CorpusCleaner',
    'CorpusSplitter',
    'split_corpus',
    'SequenceFolder',
    'fold_train',
    'fold_test',
    'RegexReplacer',
    'replace_article',
    'replace_section',
    'replace_newline',
    'replace_minutes',
    'replace_seconds',
    'PARAGRAPH_REGEX',
    'replace_single_quote',
    'replace_double_quote'
]
