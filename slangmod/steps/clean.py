from swak.funcflow.loggers import PassThroughStdOut
from swak.pd.read import ParquetReader
from swak.pd.frame import ColumnSelector
from swak.funcflow import Pipe
from ..config import config
from ..io import save_corpus, discover_books, load_books
from ..etl import (
    CorpusCleaner,
    replace_article,
    replace_section,
    replace_newline,
    replace_minutes,
    replace_seconds,
    replace_single_quote,
    replace_double_quote,
    enforce_encoding
)
from .log_messages import log_total_number_of_docs, log_total_number_of_files

LOGGER = PassThroughStdOut(__name__, config.log_level)


read_parquet = ParquetReader(config.files.raw)
select_column = ColumnSelector('text')

process_wiki40b = Pipe[[str], str](
    replace_article,
    replace_section,
    replace_newline,
    replace_minutes,
    replace_seconds,
    replace_single_quote,
    replace_double_quote,
    enforce_encoding
)

process_books = Pipe[[str], str](
    replace_minutes,
    replace_seconds,
    replace_single_quote,
    replace_double_quote,
    enforce_encoding
)

clean_wiki40b = Pipe[[str], tuple[()]](
    LOGGER.info('Starting step "clean".'),
    LOGGER.debug(f'Reading parquet files from "{config.files.raw}".'),
    read_parquet,
    select_column,
    LOGGER.info(log_total_number_of_docs),
    CorpusCleaner(process_wiki40b),
    LOGGER.debug(f'Saving *.txt files to "{config.corpus}".'),
    save_corpus,
    LOGGER.info('Finished step "clean".')
)

clean_books = Pipe[[str], tuple[()]](
    LOGGER.info('Starting step "clean".'),
    LOGGER.debug(f'Scanning "{config.files.raw}" for files.'),
    discover_books,
    LOGGER.debug(log_total_number_of_files),
    LOGGER.debug(f'Loading files from "{config.files.raw}".'),
    load_books,
    LOGGER.info(log_total_number_of_docs),
    CorpusCleaner(process_books),
    LOGGER.debug(f'Saving *.txt files to "{config.corpus}".'),
    save_corpus,
    LOGGER.info('Finished step "clean".')
)