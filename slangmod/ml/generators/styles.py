from swak.misc import ArgRepr
from ...config import config, Styles

__all__ = [
    'Style',
    'simple',
    'quote',
    'paragraph',
    'dialogue',
    'style'
]


class Style(ArgRepr):

    def __init__(self, template: str = '{} ') -> None:
        super().__init__(template)
        self.template = template

    def __call__(self, prompt: str) -> str:
        return self.template.format(prompt.strip())


simple = Style()
paragraph = Style('{}\n\n')
quote = Style('"{}" ')
dialogue = Style('"{}"\n\n')

style = {
    Styles.SIMPLE: simple,
    Styles.PARAGRAPH: paragraph,
    Styles.QUOTE: quote,
    Styles.DIALOGUE: dialogue
}[config.chat.style]
