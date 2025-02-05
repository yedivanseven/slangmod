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
    'replace_single_quote',
    'replace_double_quote'
]

# Some useful regular expressions
ARTICLE_REGEX = r'\s*_START_ARTICLE_[\s\S]*?_START_PARAGRAPH_\s*'
SECTION_REGEX = r'\s*_START_SECTION_[\s\S]*?_START_PARAGRAPH_\s*'
NEWLINE_REGEX = r'\s*_NEWLINE_\s*'
MINUTE_REGEX = r'′'
SECONDS_REGEX = r'″'


class RegexReplacer(ArgRepr):
    """Partial of python's own ``re.sub`` function.

    Parameters
    ----------
    pattern: str
        Regex pattern to match.
    repl: str or callable
        String to replace matches with or, if a callable, must accept a
        ``Match`` object (see `documentation <https://docs.python.org/3/
        library/re.html#match-objects>`__) and return a string.
    flags: int, optional
        A flag impacting the regex actions (see `documentation
        <https://docs.python.org/3/library/re.html#flags>`__).
        Default to 0, indicating no flag.
    count: int, optional
        Up to how many occurrences of `pattern` to replace. Defaults to 0,
        which results in all occurrences to be replaced.

    """

    def __init__(
            self,
            pattern: str | Pattern,
            repl: str | Callable[[Match], str],
            flags: int | RegexFlag = 0,
            count: int = 0,
    ) -> None:
        self.pattern: Pattern = re.compile(pattern, flags)
        self.repl = repl
        self.flags = flags
        self.count = count
        super().__init__(pattern, repl, flags, count)

    def __call__(self, text: str) -> str:
        """Replace matches of the cached regular expression in the text.

        Parameters
        ----------
        text: str
            The text with potential occurrence of the cached `pattern`.

        Returns
        -------
        str
            The `text` with occurrences of `pattern` replaced by `repl`.

        """
        return self.pattern.sub(self.repl, text, self.count)


# Provide ready-to-use instances for the regular expressions defined above
replace_article = RegexReplacer(ARTICLE_REGEX, '')
replace_section = RegexReplacer(SECTION_REGEX, config.tokens.eos_string)
replace_newline = RegexReplacer(NEWLINE_REGEX, config.tokens.eos_string)
replace_minutes = RegexReplacer(MINUTE_REGEX, "'")
replace_seconds = RegexReplacer(SECONDS_REGEX, '"')
replace_single_quote = RegexReplacer(r'‘|’', "'")
replace_double_quote = RegexReplacer(r'“|”', '"')
