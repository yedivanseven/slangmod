import shutil
from pathlib import Path
from itertools import compress
from pandas import Series
from swak.misc import ArgRepr
from swak.funcflow import Pipe
from swak.pd import ParquetReader, ColumnSelector, ParquetWriter
from ..config import config

__all__ = [
    'DirectoryCleaner',
    'clean_corpus_directory',
    'clean_encodings_directory',
    'write_clean_file',
    'FileTypeExtractor',
    'extract_file_type',
    'extract_file_name',
    'write_encoded_file',
    'read_column'
]


class DirectoryCleaner(ArgRepr):
    """Provide a fresh, empty target directory to write to.

    Parameters
    ----------
    folder: str
        Parent directory to clean or create. Subdirectories can be specified
        when calling instances.
    return_path: bool, optional
        Whether to return the fully resolved path to the emptied or created
        directory or not when instances are called. Defaults to ``False``

    """

    def __init__(self, folder: str, return_path: bool = False) -> None:
        self.folder = str(folder).strip()
        self.return_path = return_path
        super().__init__(self.folder)

    def __call__(self, subfolder: str = '') -> tuple[()] | str:
        """Delete the contents of an existing directory or create a new one.

        Parameters
        ----------
        subfolder: str, optional
            Subdirectory relative to the parent given at instantiation.
            Defaults to an empty string, resulting in the that parent
            directory being emptied or created.

        Returns
        -------
        tuple or str
            An empty tuple or rhe fully resolved path to the emptied or
            created directory, depending on whether `return_path` is set to
            ``False`` or ``True``.

        """
        path = Path(self.folder) / str(subfolder).strip()
        if path.exists():
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path.resolve())
            else:
                msg = 'Path "{}" seems to exist but cannot be removed!'
                raise OSError(msg.format(path.resolve()))
        path.mkdir(parents=True)
        return str(path.resolve()) if self.return_path else ()


class FileTypeExtractor(ArgRepr):
    """Determine if a file name contains one and only one of the given strings.

    Parameters
    ----------
    file_type: str
        String to test the file name for.
    *file_types: str
        Additional string to test the file name for.

    """

    def __init__(self, file_type: str, *file_types: str) -> None:
        file_type = [str(file_type).strip()]
        file_types = [str(ft).strip() for ft in file_types]
        self.types = tuple(file_type + file_types)
        super().__init__(*self.types)

    def __call__(self, path: str) -> str:
        """Determine if and, if so, which string a file name contains once.

        Parameters
        ----------
        path: str
            A (fully resolved) file name, potentially including forward
            slashed and (sub-)directories.

        Returns
        -------
        str
            The one file type of the given `file_types` that is contained in
            the stem of the file name one or more times.

        Raises
        ------
        ValueError
            If none of the cached `file_types` are contained in the stem of
            the given file name or if it contains more than one.

        """
        stem = Path(path).stem
        matches = [file_type in stem for file_type in self.types]
        match sum(matches):
            case 0:
                template = ('The type of file "{}" cannot be determined'
                            " because its name contains none of {}")
                msg = template.format(path, self.types)
            case 1:
                return self.types[matches.index(True)]
            case _:
                template = ('The type of file "{}" cannot be determined'
                            ' because it belongs to more than one: {}')
                matching_types = tuple(compress(self.types, matches))
                msg = template.format(path, matching_types)
        raise ValueError(msg)


def extract_file_name(path: str) -> str:
    """Extract the bare file name from a fully resolved path to a file.

    Parameters
    ----------
    path: str
        Fully resolved path to a file.

    Returns
    -------
    str
        The bare file name without the leading slashes and (sub-)directories.

    """
    return Path(path).name


# Provide ready-to-use instances of the DirectoryCleaner ...
clean_corpus_directory = DirectoryCleaner(config.corpus)
clean_encodings_directory = DirectoryCleaner(config.encodings)
# ... the ParquetWriter, and ...
write_clean_file = ParquetWriter(config.clean_files, create=True)
write_encoded_file = ParquetWriter(config.encoded_files, create=True)
# ... the FileTypeExtractor.
extract_file_type = FileTypeExtractor(*config.files.types)
# Because we can only ever read DataFrames, but actually would like a Series:
read_column = Pipe[[str], Series](
    ParquetReader(),
    ColumnSelector(config.files.column)
)
