from pandas import Series
from swak.funcflow.loggers import PassThroughStdOut
from swak.pd.read import ParquetReader
from swak.pd.frame import ColumnSelector
from swak.funcflow import Pipe, Map
from ..config import config
from ..io import save_corpus
from ..etl import (
    replace_article,
    replace_section,
    replace_newline,
    replace_single_quote,
    replace_double_quote,
    enforce_encoding
)
from .log_messages import log_total_number_of_docs

LOGGER = PassThroughStdOut(__name__, config.log_level)


read_parquet = ParquetReader(config.files.raw)
select_column = ColumnSelector('text')

process_wiki40b = Pipe[[str], str](
    replace_article,
    replace_section,
    replace_newline,
    replace_single_quote,
    replace_double_quote,
    enforce_encoding
)

clean_wiki40b = Pipe[[str], tuple[()]](
    LOGGER.info('Starting step "clean".'),
    LOGGER.debug(f'Reading parquet files from "{config.files.raw}"'),
    read_parquet,
    select_column,
    LOGGER.info(log_total_number_of_docs),
    Map[[str], str, Series](process_wiki40b),
    save_corpus,
    LOGGER.info('Finished step "clean".')
)
