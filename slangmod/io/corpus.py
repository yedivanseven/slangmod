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
    'CorpusFilter',
    'CorpusLoader',
    'PrefixExtractor',
    'discover_corpus',
    'discover_wiki40b',
    'discover_gutenberg',
    'discover_encodings',
    'extract_prefix',
    'extract_file_name',
    'train_filter',
    'test_filter',
    'validation_filter'
]


class CorpusDiscovery(ArgRepr):

    def __init__(
            self,
            path: str = '',
            suffix: str = 'parquet',
            sep: str = '-',
            train: str = 'train',
            test: str = 'test',
            validation: str = 'validation',
            not_found: NotFound | LiteralNotFound = NotFound.RAISE
    ) -> None:
        self.path = str(path).strip()
        self.suffix = suffix.strip(' .')
        self.sep = sep.strip()
        self.train = train.strip()
        self.test = test.strip()
        self.validation = validation.strip()
        self.not_found = str(not_found).strip().lower()
        super().__init__(
            self.path,
            self.suffix,
            self.sep,
            self.train,
            self.test,
            self.validation,
            self.not_found
        )

    @property
    def prefixes(self) -> tuple[str, str, str]:
        return self.train, self.test, self.validation

    def __call__(self, path: str = '') -> list[str]:
        path = Path(self.path) / str(path).strip()
        corpus =  [
            str(item.resolve())
            for item in path.iterdir()
            if item.is_file()
                and item.suffix == f'.{self.suffix}'
                and item.stem.split(self.sep)[0] in self.prefixes
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


class PrefixExtractor(ArgRepr):

    def __init__(self, sep: str = '-') -> None:
        self.sep = sep.strip()
        super().__init__(self.sep)

    def __call__(self, file: str) -> str:
        stem = Path(file).stem
        prefix = stem.split(self.sep)[0]
        if prefix != stem:
            return prefix
        msg = 'File name "{}" does not contain any separators "{}".'
        raise ValueError(msg.format(file, self.sep))


class CorpusFilter(ArgRepr):

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix.strip()
        super().__init__(self.prefix)

    def __call__(self, file: str) -> bool:
        return Path(file).name.startswith(self.prefix)


class CorpusLoader(ArgRepr):

    def __init__(self, reader: Callable[[str], Iterable[str]]) -> None:
        super().__init__(reader)
        self.reader = reader

    def __call__(self, files: list[str]) -> chain[str]:
        return chain.from_iterable(map(self.reader, files))


def extract_file_name(file: str) -> str:
    return Path(file).name


discover_wiki40b = CorpusDiscovery(
    config.files.wiki40b,
    config.files.suffix,
    config.files.sep,
    config.files.train,
    config.files.test,
    config.files.validation
)
discover_gutenberg = CorpusDiscovery(
    config.files.gutenberg,
    config.files.suffix,
    config.files.sep,
    config.files.train,
    config.files.test,
    config.files.validation
)
discover_corpus = CorpusDiscovery(config.corpus)
discover_encodings = CorpusDiscovery(config.encodings)

extract_prefix = PrefixExtractor(config.files.sep)

train_filter = CorpusFilter(config.files.train)
test_filter = CorpusFilter(config.files.test)
validation_filter = CorpusFilter(config.files.validation)
