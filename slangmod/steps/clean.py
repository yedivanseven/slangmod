from pandas import DataFrame
from swak.funcflow.loggers import PassThroughStdLogger
from swak.funcflow import Pipe, Map, Sum, Fork, identity
from ..config import config
from ..etl import clean_docs, trim_memory
from ..io import (
    discover_raw,
    clean_corpus_directory,
    extract_file_type,
    write_clean_file,
    read_column
)
from .log_messages import log_total_number_of_files

__all__ = ['clean']

LOGGER = PassThroughStdLogger(__name__, config.log_level)


delete_corpus_directory = Pipe[[tuple[()]], tuple[()]](
    LOGGER.debug(f'Preparing a fresh and empty folder "{config.corpus}".'),
    clean_corpus_directory
)

clean_file = Pipe[[str], tuple[()]](
    Fork[[str], tuple[DataFrame, str, str]](
        Pipe[[str], tuple[()]](
            read_column,
            clean_docs,
        ),
        extract_file_type
    ),
    write_clean_file,
    trim_memory
)

clean_files = Map[[str], tuple[()], list](clean_file)


clean = Pipe[[tuple[()]], tuple[()]](
    LOGGER.info(f'{"Resum" if config.resume else "Start"}ing step "clean".'),
    identity if config.resume else delete_corpus_directory,
    LOGGER.debug(f'Scanning "{config.files.raw}" for '
                 f'*.{config.files.suffix.strip(' .')} files.'),
    discover_raw,
    LOGGER.debug(log_total_number_of_files),
    clean_files,
    LOGGER.debug(f'Saved cleaned *.{config.files.suffix.strip(' .')}'
                 f' files to "{config.corpus}".'),
    Sum(()),  # Collapse empty tuples into one
    LOGGER.info('Finished step "clean".')
)
