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
    'paragraph_regex',
    'replace_single_quote',
    'replace_double_quote',
]


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


article_regex = r'\s*_START_ARTICLE_[\s\S]*?_START_PARAGRAPH_\s*'
section_regex = r'\s*_START_SECTION_[\s\S]*?_START_PARAGRAPH_\s*'
newline_regex = r'\s*_NEWLINE_\s*'
paragraph_regex = r'\n{2,}'

replace_article = RegexReplacer(article_regex, '')
replace_section = RegexReplacer(section_regex, config.tokens.eos_string)
replace_newline = RegexReplacer(newline_regex, config.tokens.eos_string)
replace_single_quote = RegexReplacer(r'‘|’', "'")
replace_double_quote = RegexReplacer(r'“|”', '"')
