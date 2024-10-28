from pathlib import Path
from swak.misc import ArgRepr
from ..config import config
from ..ml import Algo, tokenizer

__all__ = [
    'TokenizerSaver',
    'TokenizerLoader',
    'save_tokenizer',
    'load_tokenizer'
]


class TokenizerSaver(ArgRepr):

    def __init__(self, path: str = '') -> None:
        self.path = str(Path(path).resolve())
        super().__init__(self.path)

    def __call__(self, algo: Algo, path: str = '') -> tuple[()]:
        path = Path(self.path) / path.strip().strip(' /')
        algo.save(str(path.resolve()))
        return ()


class TokenizerLoader(ArgRepr):

    def __init__(self, algo: Algo, path: str = '') -> None:
        self.algo = algo
        self.path = str(Path(path).resolve())
        super().__init__(self.path)

    def __call__(self, path: str = '') -> Algo:
        path = Path(self.path) / path.strip().strip(' /')
        return self.algo.from_file(str(path.resolve()))


save_tokenizer = TokenizerSaver(config.files.tokenizer)
load_tokenizer = TokenizerLoader(tokenizer, config.files.tokenizer)
