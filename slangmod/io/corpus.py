import warnings
import random
from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path
from tqdm import tqdm
from swak.misc import ArgRepr
from swak.text import NotFound, LiteralNotFound
from ..config import config

__all__ = [
    'NotFound',
    'CorpusDiscovery',
    'CorpusLoader',
    'CorpusSaver',
    'discover_corpus',
    'discover_wiki40b',
    'discover_gutenberg',
    'load_corpus',
    'load_books',
    'save_corpus',
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

    def __init__(
            self,
            shuffle: bool = True,
            not_found: NotFound | LiteralNotFound = NotFound.RAISE
    ) -> None:
        self.shuffle = shuffle
        self.not_found = str(not_found).strip().lower()
        super().__init__(shuffle, self.not_found)

    @staticmethod
    def jumble(files: list[str]) -> list[str]:
        return random.sample(files, len(files))

    def __call__(self, files: list[str]) -> list[str]:
        actual_files = self.jumble(files) if self.shuffle else files
        corpus = []
        for file in tqdm(actual_files, 'Documents', leave=False):
            try:
                with Path(file).open() as stream:
                    text = stream.read()
            except FileNotFoundError as error:
                match self.not_found:
                    case NotFound.IGNORE:
                        continue
                    case NotFound.WARN:
                        msg = f'File "{file}" not found. Moving on ...'
                        warnings.warn(msg)
                        continue
                    case NotFound.RAISE:
                        raise error
            else:
                corpus.append(text.strip())
        if corpus:
            return corpus
        msg = 'No files to load!'
        match self.not_found:
            case NotFound.WARN:
                warnings.warn(msg)
            case NotFound.RAISE:
                raise FileNotFoundError(msg)
        return corpus


class CorpusSaver(ArgRepr):

    def __init__(self, path: str = '', create: bool = False) -> None:
        self.path = str(path).strip()
        self.create = create
        super().__init__(self.path, create)

    def __call__(self, corpus: Iterable[str], *parts: str) -> tuple[()]:
        path = Path(self.path.format(*parts).strip())
        if self.create:
            path.mkdir(parents=True, exist_ok=True)
        for item in tqdm(corpus, 'Documents', leave=False):
            name = sha256(item.encode()).hexdigest()
            file = path / f'{name}.txt'
            with file.open('wt') as stream:
                stream.write(item)
        return ()

discover_wiki40b = CorpusDiscovery(config.files.wiki40b)
discover_gutenberg = CorpusDiscovery(config.files.gutenberg)
discover_corpus = CorpusDiscovery(config.corpus)

load_books = CorpusLoader(config.data.shuffle)
load_corpus = CorpusLoader(config.data.shuffle)
save_corpus = CorpusSaver(config.corpus, create=True)
