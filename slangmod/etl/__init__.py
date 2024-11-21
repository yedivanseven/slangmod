from .encoding import EncodingEnforcer, enforce_encoding
from .terminator import Terminator, terminate
from .splitter import CorpusSplitter, split_corpus
from .folder import SequenceFolder, fold_train, fold_test
from .regex import (
    RegexReplacer,
    replace_article,
    replace_section,
    replace_newline,
    paragraph_regex,
    replace_paragraph,
    replace_single_quote,
    replace_double_quote
)

__all__ = [
    'EncodingEnforcer',
    'enforce_encoding',
    'Terminator',
    'terminate',
    'CorpusSplitter',
    'split_corpus',
    'SequenceFolder',
    'fold_train',
    'fold_test',
    'RegexReplacer',
    'replace_article',
    'replace_section',
    'replace_newline',
    'paragraph_regex',
    'replace_paragraph',
    'replace_single_quote',
    'replace_double_quote'
]
