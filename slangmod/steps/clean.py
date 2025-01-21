from pandas import DataFrame
from swak.funcflow.loggers import PassThroughStdOut
from swak.funcflow import Pipe, Map, Sum, Fork
from ..config import config
from ..io import (
    discover_wiki40b,
    discover_gutenberg,
    clean_corpus_directory,
    extract_file_type,
    write_clean_file,
    read_column
)
from ..etl import (
    CorpusCleaner,
    replace_article,
    replace_section,
    replace_newline,
    replace_minutes,
    replace_seconds,
    replace_single_quote,
    replace_double_quote,
    enforce_encoding,
    trim_memory
)
from .log_messages import log_total_number_of_files

__all__ = ['clean']

LOGGER = PassThroughStdOut(__name__, config.log_level)


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
process_wiki40b_docs = CorpusCleaner(
    wiki40b_processor,
    config.data.jitter,
    'Documents'
)
process_wiki40b_file = Pipe[[str], tuple[()]](
    Fork[[str], tuple[str, DataFrame, str]](
        Pipe[[str], tuple[()]](
            read_column,
            process_wiki40b_docs,
        ),
        extract_file_type
    ),
    write_clean_file,
    trim_memory
)
process_wiki40b = Map[[str], tuple[()], list](process_wiki40b_file)
clean_wiki40b = Pipe[[tuple[()]], tuple[()]](
    LOGGER.debug(f'Scanning "{config.files.wiki40b}" for *.parquet files.'),
    discover_wiki40b,
    LOGGER.debug(log_total_number_of_files),
    process_wiki40b,
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
process_gutenberg_docs = CorpusCleaner(
    gutenberg_processor,
    config.data.jitter,
    'Documents'
)
process_gutenberg_file = Pipe[[str], tuple[()]](
    Fork[[str], tuple[str, DataFrame, str]](
        Pipe[[str], tuple[()]](
            read_column,
            process_gutenberg_docs,
        ),
        extract_file_type,
    ),
    write_clean_file,
    trim_memory
)
process_gutenberg = Map[[str], tuple[()], list](process_gutenberg_file)
clean_gutenberg = Pipe[[tuple[()]], tuple[()]](
    LOGGER.debug(f'Scanning "{config.files.gutenberg}" for *.parquet files.'),
    discover_gutenberg,
    LOGGER.debug(log_total_number_of_files),
    process_gutenberg,
    LOGGER.debug(f'Saved cleaned *.parquet files to "{config.corpus}".'),
    Sum(()),
)

clean = Pipe[[tuple[()]], tuple[()]](
    LOGGER.info('Starting step "clean".'),
LOGGER.debug(f'Preparing a fresh and empty folder "{config.corpus}".'),
    clean_corpus_directory,
    Fork[[tuple[()]], tuple[()]](
        clean_wiki40b,
        # clean_gutenberg
    ),
    LOGGER.info('Finished step "clean".')
)
