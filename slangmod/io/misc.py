import shutil
from typing import Any
from pathlib import Path
from pandas import Series
from swak.misc import ArgRepr
from swak.funcflow import Pipe
from swak.pd import ParquetReader, ColumnSelector
from ..config import config

__all__ = [
    'DirectoryCleaner',
    'clean_corpus_directory',
    'clean_encodings_directory',
    'FileTypeExtractor',
    'extract_file_type',
    'extract_file_name',
    'read_column'
]


class DirectoryCleaner(ArgRepr):

    def __init__(self, folder: str) -> None:
        self.folder = folder.strip()
        super().__init__(self.folder)

    def __call__(self, *_: Any, **__: Any) -> tuple[()]:
        path = Path(self.folder)
        if path.exists():
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path.resolve())
            else:
                msg = 'Path "{}" seems to exist but cannot be removed!'
                raise OSError(msg.format(path.resolve()))
        path.mkdir(parents=True)
        return ()


# ToDo: Do we really need the separator? maybe string contain is enough?
class FileTypeExtractor(ArgRepr):

    def __init__(self, sep: str, file_type: str, *file_types: str) -> None:
        self.sep = sep.strip()
        self.types = {t.strip().lower() for t in file_types + (file_type,)}
        super().__init__(self.sep, *self.types)

    def __call__(self, file: str) -> str:
        stem = Path(file).stem
        parts = {part.lower() for part in stem.split(self.sep)}
        matches = tuple(self.types & parts)
        match matches:
            case file_type,:
                return file_type
            case _, __, *___:
                template = ('The type of file "{}" cannot be determined'
                            ' because it belongs to more than one: {}')
                msg = template.format(file, matches)
            case _:
                template = ('The type of file "{}" cannot be determined'
                            ' because it contains none of {}')
                msg = template.format(file, tuple(self.types))
        raise ValueError(msg)


def extract_file_name(file: str) -> str:
    return Path(file).name


clean_corpus_directory = DirectoryCleaner(config.corpus)
clean_encodings_directory = DirectoryCleaner(config.encodings)

extract_file_type = FileTypeExtractor(config.files.sep, *config.files.types)

read_column = Pipe[[str], Series](
    ParquetReader(),
    ColumnSelector(config.files.column)
)
