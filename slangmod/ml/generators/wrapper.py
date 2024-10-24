from swak.misc import ArgRepr
from ...config import config, Wrappers

__all__ = [
    'Wrapper',
    'simple',
    'quote',
    'paragraph',
    'dialogue',
    'wrapper'
]


class Wrapper(ArgRepr):

    def __init__(self, template: str = '{} ') -> None:
        super().__init__(template)
        self.template = template

    def __call__(self, prompt: str) -> str:
        return self.template.format(prompt.strip())


simple = Wrapper()
paragraph = Wrapper('{}\n\n')
quote = Wrapper('"{}" ')
dialogue = Wrapper('"{}"\n\n')

wrapper = {
    Wrappers.SIMPLE: simple,
    Wrappers.PARAGRAPH: paragraph,
    Wrappers.QUOTE: quote,
    Wrappers.DIALOGUE: dialogue
}[config.chat.wrapper]
