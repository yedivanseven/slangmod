from pandas import Series, DataFrame
from swak.funcflow import apply, Fork, Pipe, Route, Sum, Map
from swak.funcflow.loggers import PassThroughStdLogger
from ..etl import to_frame, trim_memory
from ..config import config
from ..ml import Algo
from .log_messages import log_total_number_of_files, log_encode_file
from ..io import (
    load_tokenizer,
    clean_encodings_directory,
    discover_corpus,
    read_column,
    extract_file_name,
    write_encoded_file
)

__all__ = ['encode']

LOGGER = PassThroughStdLogger(__name__, config.log_level)


load_column = Pipe[[str], Series](
    LOGGER.debug(log_encode_file),
    read_column
)

load = Route[[str], tuple[Algo, Series]](
    [(), 0],
    load_tokenizer,
    load_column
)

encode_file = Pipe[[str], tuple[()]](
    Fork[[str], tuple[()]](
        Pipe[[str], DataFrame](
            load,
            apply,
            trim_memory,
            to_frame
        ),
        extract_file_name
    ),
    write_encoded_file,
    trim_memory,
)

encode = Pipe[[tuple[()]], tuple[()]](
    LOGGER.info('Starting step "encode".'),
    LOGGER.debug(f'Preparing a fresh and empty folder "{config.encodings}".'),
    clean_encodings_directory,
    LOGGER.debug(f'Scanning "{config.files.corpus}" for '
                 f'*.{config.files.suffix.strip(' .')} files.'),
    discover_corpus,
    LOGGER.debug(log_total_number_of_files),
    LOGGER.debug(f'Loading tokenizer from file "{config.tokenizer_file}".'),
    Map[[str], tuple[()], list](encode_file),
    LOGGER.debug(f'Saved encoded corpus to "{config.encodings}".'),
    Sum(()),  # Collapse empty tuples into one
    LOGGER.info('Finished step "encode".'),
)
