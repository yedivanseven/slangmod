import re
from swak.misc import ArgRepr
from ..config import config

__all__ = [
    'EncodingEnforcer',
    'enforce_encoding'
]


class EncodingEnforcer(ArgRepr):

    pattern = re.compile(r'(?:\\N{.+?})')

    def __init__(self, encoding: str, repl: str = ' ', count: int = 0) -> None:
        self.encoding = encoding
        self.repl = repl
        self.count = count
        super().__init__(encoding, repl, count)

    def __call__(self, text: str, count: int | None = None) -> str:
        count = self.count if count is None else count
        replaced = text.encode(
            self.encoding,
            'namereplace'
        ).decode(self.encoding)
        return self.pattern.sub(self.repl, replaced, count)


enforce_encoding = EncodingEnforcer(config.tokens.encoding)
