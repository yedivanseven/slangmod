import warnings
from pathlib import Path
from swak.misc import ArgRepr
from swak.text import NotFound
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
            not_found: str = NotFound.RAISE
    ) -> None:
        self.path = str(Path(path).resolve())
        self.not_found = str(not_found)
        super().__init__(self.path, self.not_found)

    def __call__(self, path: str = '') -> list[str]:
        path = Path(self.path) / path.strip(' /')
        corpus =  [
            str(item.resolve())
            for item in path.iterdir()
            if item.is_file() and item.suffix == '.txt'
        ] if path.exists() and path.is_dir() else []
        if corpus:
            return corpus
        msg = 'No *.txt files found in folder {}!'
        match self.not_found:
            case NotFound.WARN:
                warnings.warn(msg.format(path.resolve()))
            case NotFound.RAISE:
                raise FileNotFoundError(msg.format(path.resolve()))
        return corpus


class CorpusLoader(ArgRepr):

    def __init__(self, not_found: str = NotFound.RAISE) -> None:
        self.not_found = str(not_found)
        super().__init__(self.not_found)

    @staticmethod
    def read(file: str) -> str:
        with Path(file).open() as stream:
            text = stream.read()
        return text.strip()

    def __call__(self, files: list[str]) -> str:
        corpus = ' [EOS] '.join(map(self.read, files))
        if corpus:
            return corpus
        msg = 'No corpus to load!'
        match self.not_found:
            case NotFound.WARN:
                warnings.warn(msg)
            case NotFound.RAISE:
                raise FileNotFoundError(msg)
        return corpus


discover_corpus = CorpusDiscovery(config.corpus, NotFound.WARN)
load_corpus = CorpusLoader(NotFound.RAISE)
