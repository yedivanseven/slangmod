from pandas import Series, DataFrame
from swak.pd import ParquetWriter
from swak.funcflow import apply, Fork, Pipe, Route, Sum, Map
from swak.funcflow.loggers import PassThroughStdOut
from ..etl import to_frame, trim_memory
from ..config import config
from ..ml import Algo
from .log_messages import log_total_number_of_files, log_encode_file
from ..io import (
    load_tokenizer,
    discover_corpus,
    read_column,
    extract_file_name,
    save_config
)

LOGGER = PassThroughStdOut(__name__, config.log_level)

write_parquet = ParquetWriter(config.encodings + '/{}', create=True)

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
    write_parquet,
    trim_memory,
)

encode = Pipe[[tuple[()]], tuple[()]](
    LOGGER.debug(f'Saving config file to "{config.config_file}".'),
    save_config,
    LOGGER.info('Starting step "encode".'),
    LOGGER.debug(f'Scanning "{config.corpus}" for files.'),
    LOGGER.debug(f'Writing to "{config.encodings}".'),
    discover_corpus,
    LOGGER.debug(log_total_number_of_files),
    Map[[str], tuple[()], list](encode_file),
    Sum(()),
    LOGGER.info('Finished step "encode".'),
)
