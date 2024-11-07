from collections.abc import Iterable
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

    def __init__(
            self,
            template: str = '{} ',
            char: str | Iterable[str] = '"',
            *chars: str
    ) -> None:
        self.template = template
        self.chars = ((char,) if isinstance(char, str) else char) + chars
        super().__init__(template, *chars)

    def strip(self, prompt: str) -> str:
        naked = prompt.strip()
        for char in self.chars:
            naked = naked.strip(char)
        return naked.strip()

    def __call__(self, prompt: str) -> str:
        return self.template.format(self.strip(prompt))


space = Style('{} ')
paragraph = Style('{}' + f'{config.tokens.eos_string}')
quote = Style('"{}," ', ",")
dialogue = Style('"{}"' + f'{config.tokens.eos_string}')

style = {
    Styles.SPACE: space,
    Styles.PARAGRAPH: paragraph,
    Styles.QUOTE: quote,
    Styles.DIALOGUE: dialogue
}[config.chat.style]
