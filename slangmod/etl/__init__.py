"""Tools to clean, preprocess, and re-format your text corpus."""

from swak.funcflow import Pipe
from swak.dictionary import ValuesGetter
from ..config import config, Cleaners
from .encoding import EncodingEnforcer, enforce_encoding
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
    'EncodingEnforcer',
    'enforce_encoding',
    'CorpusCleaner',
    'ToFrame',
    'to_frame',
    'MemoryTrimmer',
    'trim_memory',
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

# Assemble the document-cleaning pipeline according to the config.
quotes = Pipe[[str], [str]](
    replace_minutes,
    replace_seconds,
    replace_single_quote,
    replace_double_quote
)
wiki40b = Pipe[[str], str](
    replace_article,
    replace_section,
    replace_newline
)
cleaners = {
    Cleaners.QUOTES: quotes,
    Cleaners.WIKI40B: wiki40b,
    Cleaners.ENCODING: enforce_encoding
}
processor = Pipe[[str], str](*ValuesGetter(config.files.cleaners)(cleaners))
