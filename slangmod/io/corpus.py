import warnings
from collections.abc import Iterable, Callable
from itertools import chain
from pathlib import Path
from swak.misc import ArgRepr
from swak.text import NotFound, LiteralNotFound
from swak.funcflow import Filter
from ..config import config
from .files import read_column

__all__ = [
    'NotFound',
    'CorpusDiscovery',
    'CorpusFilter',
    'CorpusLoader',
    'load_corpus',
    'discover_raw',
    'discover_corpus',
    'discover_encodings',
    'filter_train_files',
    'filter_test_files',
    'filter_validation_files'
]


class CorpusDiscovery(ArgRepr):
    """Discover files in a given directory and filter by name and suffix.

    Parameters
    ----------
    folder: str, optional
        Parent directory to search for files. Subdirectories can be specified
        when calling instances. Defaults to the working directory of the
        current python interpreter.
    *file_types: str, optional
        File names must contain at least one of these strings.
    suffix: str, optional
        Extension glob pattern that files must match (without leading dot).
        Defaults to "parquet".
    not_found: str, optional
        What to do if either the directory does not exist or no matching files
        are found in the given directory. One of "ignore", "warn", or "raise".
        Use the `NotFound <https://yedivanseven.github.io/swak/text.html#swak.
        text.misc.NotFound>`_ enum to avoid typos. Defaults to "raise".
        If set otherwise, an empty tuple of file names might be returned.

    """

    def __init__(
            self,
            folder: str = '',
            *file_types: str,
            suffix: str = 'parquet',
            not_found: NotFound | LiteralNotFound = NotFound.RAISE,
    ) -> None:
        self.folder = str(folder).strip()
        self.types = tuple(
            str(file_type).strip() for file_type in file_types
        ) if file_types else ('',)
        self.suffix = str(suffix).strip(' .')
        self.not_found = str(not_found).strip().lower()
        super().__init__(
            self.folder,
            *self.types,
            suffix=self.suffix,
            not_found=self.not_found
        )

    def __call__(self, subfolder: str = '') -> list[str]:
        """Chose subdirectory and filter names of files found therein.

        Parameters
        ----------
        subfolder: str, optional
            Subdirectory relative to the parent given at instantiation.
            Defaults to an empty string, resulting in the that parent
            directory to be searched.

        Returns
        -------
        list
            Fully resolved names of files that match the given criteria
            from within the specified directory.

        Raises
        ------
        FileNotFoundError
            Only if `not_found` is set to "raise", and then only if either the
            directory was not found or no files matching the specified criteria
            were found in that directory.

        """
        path = Path(self.folder) / str(subfolder).strip()
        corpus = [
            str(item.resolve())
            for item in path.glob(f'*.{self.suffix}')
            if item.is_file()
            and any(prefix in item.name for prefix in self.types)
        ] if path.exists() and path.is_dir() else []
        if corpus:
            return corpus
        # If no files were found, act according to the not_found flag
        template = 'No *.{} files with any of {} in their name in folder "{}"!'
        msg = template.format(self.suffix, self.types, path.resolve())
        match self.not_found:
            case NotFound.WARN:
                warnings.warn(msg)
            case NotFound.RAISE:
                raise FileNotFoundError(msg)
        return corpus


class CorpusLoader(ArgRepr):
    """Read files with multiple documents and provide an iterator over all.

    Parameters
    ----------
    reader: callable
        Must return some sort of iterable over documents (=strings), when
        given a file name.

    """

    def __init__(self, reader: Callable[[str], Iterable[str]]) -> None:
        super().__init__(reader)
        self.reader = reader

    def __call__(self, files: Iterable[str]) -> chain[str]:
        """Read files with multiple documents and provide an iterator over all.

        Parameters
        ----------
        files: iterable over str
            Names of files to chain documents from.

        Returns
        -------
        Iterator
            An ``itertools.chain`` iterator over all documents from all files.

        """
        return chain.from_iterable(map(self.reader, files))


class CorpusFilter(ArgRepr):
    """Determine whether a given string is part of a fully resolved file name.

    Parameters
    ----------
    part: str
        Part of the file name to filter for.

    """

    def __init__(self, part: str) -> None:
        self.part = part
        super().__init__(self.part)

    def __call__(self, file: str) -> bool:
        """Determine whether the cached string is part of the file name.

        Parameters
        ----------
        file: str
            Name of the file to test. Can include parent folder(s).

        Returns
        -------
        bool
            Whether the cached `part` occurs in the file name at least once.

        """
        return self.part in Path(file).name


# Provide ready-to-use instances of the CorpusDiscovery
discover_raw = CorpusDiscovery(
    config.files.raw,
    *config.files.types,
    suffix=config.files.suffix,
    not_found=NotFound.RAISE
)
discover_corpus = CorpusDiscovery(
    config.corpus,
    *config.files.types,
    suffix=config.files.suffix,
    not_found=NotFound.RAISE
)
discover_encodings = CorpusDiscovery(
    config.encodings,
    *config.files.types,
    suffix=config.files.suffix,
    not_found=NotFound.RAISE
)

# Provide a ready-to-use instance of the CorpusLoader
load_corpus = CorpusLoader(read_column)

# Provide ready-to-use instances of the CorpusFilter ...
train_files_filter = CorpusFilter(config.files.train)
test_files_filter = CorpusFilter(config.files.test)
validation_files_filter = CorpusFilter(config.files.validation)
# ... and the actual filters
filter_train_files = Filter[str, list](train_files_filter)
filter_test_files = Filter[str, list](test_files_filter)
filter_validation_files = Filter[str, list](validation_files_filter)
