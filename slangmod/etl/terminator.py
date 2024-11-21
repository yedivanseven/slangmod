from swak.misc import ArgRepr
from ..config import config

__all__ = [
    'Terminator',
    'terminate'
]


class Terminator(ArgRepr):

    def __init__(self, eos_repl: str) -> None:
        self.eos_repl = eos_repl
        super().__init__(self.eos_repl)

    def __call__(self, text: str) -> str:
        return text.rstrip() + self.eos_repl


terminate = Terminator(config.tokens.eos_repl)
