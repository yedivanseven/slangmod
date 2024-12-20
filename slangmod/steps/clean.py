from pandas import DataFrame
from swak.funcflow.loggers import PassThroughStdOut
from swak.pd import ParquetReader, ParquetWriter, ColumnSelector
from swak.funcflow import Pipe, Map, Sum, Fork, Route
from ..config import config
from ..io import discover_wiki40b, discover_gutenberg, extract_prefix
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
from .log_messages import log_total_number_of_files

LOGGER = PassThroughStdOut(__name__, config.log_level)
TARGET = config.corpus + '/{}' + config.files.sep + '{}.' + config.files.suffix

read_parquet = ParquetReader()
write_parquet = ParquetWriter(TARGET, create=True)
select_column = ColumnSelector(config.files.column)

wiki40b_processor = Pipe[[str], str](
    replace_article,
    replace_section,
    replace_newline,
    replace_minutes,
    replace_seconds,
    replace_single_quote,
    replace_double_quote,
    enforce_encoding
)
process_wiki40b_docs = CorpusCleaner(wiki40b_processor, 'Documents')
process_wiki40b_file = Pipe[[str], tuple[()]](
    Fork[[str], tuple[str, DataFrame, str]](
        extract_prefix,
        Pipe[[str], tuple[()]](
            read_parquet,
        select_column,
            process_wiki40b_docs,
        )
    ),
    Route[[str, DataFrame, str], tuple[()]]([(1, 0, 2)], write_parquet)
)
process_wiki40b_corpus = Map[[str], tuple[()], list](process_wiki40b_file)
clean_wiki40b = Pipe[[tuple[()]], tuple[()]](
    LOGGER.debug(f'Scanning "{config.files.wiki40b}" for *.parquet files.'),
    discover_wiki40b,
    LOGGER.debug(log_total_number_of_files),
    process_wiki40b_corpus,
    LOGGER.debug(f'Saved cleaned *.parquet files to "{config.corpus}".'),
    Sum(()),
)

gutenberg_processor = Pipe[[str], str](
    replace_minutes,
    replace_seconds,
    replace_single_quote,
    replace_double_quote,
    enforce_encoding
)
process_gutenberg_docs = CorpusCleaner(gutenberg_processor, 'Documents')
process_gutenberg_file = Pipe[[str], tuple[()]](
    Fork[[str], tuple[str, DataFrame, str]](
        extract_prefix,
        Pipe[[str], tuple[()]](
            read_parquet,
        select_column,
            process_gutenberg_docs,
        )
    ),
    Route[[str, DataFrame, str], tuple[()]]([(1, 0, 2)], write_parquet)
)
process_gutenberg_corpus = Map[[str], tuple[()], list](process_gutenberg_file)
clean_gutenberg = Pipe[[tuple[()]], tuple[()]](
    LOGGER.debug(f'Scanning "{config.files.gutenberg}" for *.parquet files.'),
    discover_gutenberg,
    LOGGER.debug(log_total_number_of_files),
    process_gutenberg_corpus,
    LOGGER.debug(f'Saved cleaned *.parquet files to "{config.corpus}".'),
    Sum(()),
)

clean = Pipe[[tuple[()]], tuple[()]](
    LOGGER.info('Starting step "clean".'),
    Fork[[tuple[()]], tuple[()]](
        clean_wiki40b,
        clean_gutenberg
    ),
    LOGGER.info('Finished step "clean".')
)
