import re
from re import Match, Pattern, RegexFlag
from collections.abc import Callable
from swak.misc import ArgRepr
from ..config import config

__all__ = [
    'RegexReplacer',
    'replace_article',
    'replace_section',
    'replace_newline',
    'replace_minutes',
    'replace_seconds',
    'PARAGRAPH_REGEX',
    'replace_single_quote',
    'replace_double_quote',
]

ARTICLE_REGEX = r'\s*_START_ARTICLE_[\s\S]*?_START_PARAGRAPH_\s*'
SECTION_REGEX = r'\s*_START_SECTION_[\s\S]*?_START_PARAGRAPH_\s*'
NEWLINE_REGEX = r'\s*_NEWLINE_\s*'
MINUTE_REGEX = r'′'
SECONDS_REGEX = r'″'
PARAGRAPH_REGEX = r'\n{2,}'


class RegexReplacer(ArgRepr):

    def __init__(
            self, pattern: str | Pattern,
            repl: str | Callable[[Match], str],
            flags: int | RegexFlag = 0,
            count: int = 0,
    ) -> None:
        self.pattern = re.compile(pattern, 0 if flags is None else flags)
        self.repl = repl
        self.flags = flags
        self.count = count
        super().__init__(pattern, repl, flags, count)

    def __call__(
            self,
            text: str,
            count: int | None = None,
            flags: int | RegexFlag | None = None
    ) -> str:
        count = self.count if count is None else count
        flags = self.flags if flags is None else flags
        return re.sub(self.pattern, self.repl, text, count, flags)


replace_article = RegexReplacer(ARTICLE_REGEX, '')
replace_section = RegexReplacer(SECTION_REGEX, config.tokens.eos_string)
replace_newline = RegexReplacer(NEWLINE_REGEX, config.tokens.eos_string)
replace_minutes = RegexReplacer(MINUTE_REGEX, "'")
replace_seconds = RegexReplacer(SECONDS_REGEX, '"')
replace_single_quote = RegexReplacer(r'‘|’', "'")
replace_double_quote = RegexReplacer(r'“|”', '"')
