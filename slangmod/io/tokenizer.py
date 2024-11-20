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

    def __init__(self, path: str = '', create: bool = False) -> None:
        self.path = str(path).strip()
        self.create = create
        super().__init__(self.path)

    def __call__(self, algo: Algo, *parts: str) -> tuple[()]:
        path = Path(self.path.format(*parts).strip())
        file = str(path.resolve())
        if self.create:
            path.parent.mkdir(parents=True, exist_ok=True)
        algo.save(file)
        return ()


class TokenizerLoader(ArgRepr):

    def __init__(self, algo: Algo, path: str = '') -> None:
        self.algo = algo
        self.path = str(path).strip()
        super().__init__(self.path)

    def __call__(self, path: str = '') -> Algo:
        path = Path(self.path) / str(path).strip()
        file = str(path.resolve())
        return self.algo.from_file(file)


save_tokenizer = TokenizerSaver(config.tokenizer_file, True)
load_tokenizer = TokenizerLoader(tokenizer, config.tokenizer_file)
