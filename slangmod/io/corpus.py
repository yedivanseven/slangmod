import warnings
from collections.abc import Iterable, Callable
from itertools import chain
from pathlib import Path
from swak.misc import ArgRepr
from swak.text import NotFound, LiteralNotFound
from ..config import config

__all__ = [
    'NotFound',
    'CorpusDiscovery',
    'CorpusLoader',
    'discover_corpus',
    'discover_wiki40b',
    'discover_gutenberg'
]


class CorpusDiscovery(ArgRepr):

    def __init__(
            self,
            path: str = '',
            suffix: str = 'parquet',
            not_found: NotFound | LiteralNotFound = NotFound.RAISE
    ) -> None:
        self.path = str(path).strip()
        self.suffix = suffix.strip(' .')
        self.not_found = str(not_found).strip().lower()
        super().__init__(self.path, self.suffix, self.not_found)

    def __call__(self, path: str = '') -> list[str]:
        path = Path(self.path) / str(path).strip()
        corpus =  [
            str(item.resolve())
            for item in path.iterdir()
            if item.is_file() and item.suffix == f'.{self.suffix}'
        ] if path.exists() and path.is_dir() else []
        if corpus:
            return corpus
        msg = 'No *.{} files found in folder "{}"!'
        interpolated = msg.format(self.suffix, path.resolve())
        match self.not_found:
            case NotFound.WARN:
                warnings.warn(interpolated)
            case NotFound.RAISE:
                raise FileNotFoundError(interpolated)
        return corpus


class CorpusLoader(ArgRepr):

    def __init__(self, reader: Callable[[str], Iterable[str]]) -> None:
        super().__init__(reader)
        self.reader = reader

    def __call__(self, files: list[str]) -> chain[str]:
        return chain.from_iterable(map(self.reader, files))


discover_wiki40b = CorpusDiscovery(config.files.wiki40b)
discover_gutenberg = CorpusDiscovery(config.files.gutenberg)
discover_corpus = CorpusDiscovery(config.corpus)
