from pathlib import Path
from tokenizers import Tokenizer
from swak.misc import ArgRepr
from ..config import config

__all__ = [
    'TokenizerSaver',
    'TokenizerLoader',
    'save_tokenizer',
    'load_tokenizer'
]


class TokenizerSaver(ArgRepr):

    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.path = str(Path(path).resolve())

    def __call__(self, tokenizer: Tokenizer) -> tuple[()]:
        tokenizer.save(self.path)
        return ()


class TokenizerLoader(ArgRepr):

    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.path = str(Path(path).resolve())

    def __call__(self) -> Tokenizer:
        return Tokenizer.from_file(self.path)


save_tokenizer = TokenizerSaver(config.tokenizer_file)
load_tokenizer = TokenizerLoader(config.tokenizer_file)
