import warnings
import random
from pathlib import Path
from swak.misc import ArgRepr
from swak.text import NotFound  # ToDo: Import also LiteralNotFound!
from ..config import config

__all__ = [
    'NotFound',
    'CorpusDiscovery',
    'CorpusLoader',
    'discover_corpus',
    'load_corpus'
]


class CorpusDiscovery(ArgRepr):

    def __init__(
            self,
            path: str = '',
            not_found: str = NotFound.RAISE  # ToDo LiteralNotFound!
    ) -> None:
        self.path = path.strip()
        self.not_found = str(not_found)
        super().__init__(self.path, self.not_found)

    def __call__(self, path: str = '') -> list[str]:
        path = Path(self.path) / path.strip().strip(' /')
        corpus =  [
            str(item.resolve())
            for item in path.iterdir()
            if item.is_file() and item.suffix == '.txt'
        ] if path.exists() and path.is_dir() else []
        if corpus:
            return corpus
        msg = 'No *.txt files found in folder "{}"!'
        match self.not_found:
            case NotFound.WARN:
                warnings.warn(msg.format(path.resolve()))
            case NotFound.RAISE:
                raise FileNotFoundError(msg.format(path.resolve()))
        return corpus


class CorpusLoader(ArgRepr):

    def __init__(
            self,
            eos_symbol: str,
            shuffle: bool = True,
            not_found: str = NotFound.RAISE  # ToDo LiteralNotFound!
    ) -> None:
        self.eos_symbol = eos_symbol
        self.shuffle = shuffle
        self.not_found = str(not_found).strip().lower()
        super().__init__(eos_symbol, shuffle, self.not_found)

    @property
    def sep(self) -> str:
        return f' {self.eos_symbol} '

    @property
    def end(self) -> str:
        return f' {self.eos_symbol}'

    @staticmethod
    def jumble(files: list[str]) -> list[str]:
        return random.sample(files, len(files))

    @staticmethod
    def read(file: str) -> str:
        with Path(file).open() as stream:
            text = stream.read()
        return text.strip()

    def __call__(self, files: list[str]) -> str:
        files = self.jumble(files) if self.shuffle else files
        corpus = self.sep.join(map(self.read, files)) + self.end
        if corpus:
            return corpus
        msg = 'No corpus to load!'
        match self.not_found:
            case NotFound.WARN:
                warnings.warn(msg)
            case NotFound.RAISE:
                raise FileNotFoundError(msg)
        return corpus


discover_corpus = CorpusDiscovery(config.corpus)
load_corpus = CorpusLoader(config.tokens.eos_symbol, config.data.shuffle)
