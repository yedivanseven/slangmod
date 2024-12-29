from .duplicates import DuplicateDropper, drop_duplicates
from .encoding import EncodingEnforcer, enforce_encoding
from .folder import SequenceFolder, fold_train, fold_test
from .cleaner import CorpusCleaner
from .frame import ToFrame, to_frame
from .memory import MemoryTrimmer, trim_memory
from .shuffle import Shuffle
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
    'DuplicateDropper',
    'drop_duplicates',
    'EncodingEnforcer',
    'enforce_encoding',
    'CorpusCleaner',
    'ToFrame',
    'to_frame',
    'MemoryTrimmer',
    'trim_memory',
    'SequenceFolder',
    'fold_train',
    'fold_test',
    'Shuffle',
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
