from pathlib import Path
from pandas import Series
from swak.misc import ArgRepr
from swak.funcflow import Pipe
from swak.pd import ParquetReader, ColumnSelector
from ..config import config


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


def extract_file_name(file: str) -> str:
    return Path(file).name


extract_prefix = PrefixExtractor(config.files.sep)
read_column = Pipe[[str], Series](
    ParquetReader(),
    ColumnSelector(config.files.column)
)
