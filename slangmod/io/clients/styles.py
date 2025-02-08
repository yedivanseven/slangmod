from swak.misc import ArgRepr
from ...config import config, Styles

__all__ = [
    'Style',
    'space',
    'quote',
    'paragraph',
    'dialogue',
    'style'
]


class Style(ArgRepr):
    """Format user prompt into a template to encourage model answer styles.

    Parameters
    ----------
    template: str, optional
        A python string containing a single pair of curly brackets where the
        user prompt will go. Defaults to "{} ".
    strip: str, optional
        Prior to being inserted into the `template`, the user prompt will be
        stripped of all word-delimitation characters and, additionally, of all
        the characters in the given string, both left and right.
        Defaults to ``None``, which results in only word-delimitation
        characters being stripped.

    """
    def __init__(
            self,
            template: str = '{} ',
            strip: str | None = None
    ) -> None:
        self.template = template
        self.strip = strip
        super().__init__(template, strip)

    def __call__(self, prompt: str) -> str:
        """Format user prompt by stripping characters and using a template.

        Parameters
        ----------
        prompt: str
            The user prompt.

        Returns
        -------
        str
            The formatted user prompt.

        """
        return self.template.format(prompt.strip().strip(self.strip))


# Provide a few meaningful example instances of Style ...
space = Style('{} ')
paragraph = Style('{} ' + config.tokens.eos_symbol)
quote = Style('"{}," ', '",')
dialogue = Style('"{}" ' + config.tokens.eos_symbol, '"')
# ... to select from via the project config.
style = {
    Styles.SPACE: space,
    Styles.PARAGRAPH: paragraph,
    Styles.QUOTE: quote,
    Styles.DIALOGUE: dialogue
}[config.chat.style]
