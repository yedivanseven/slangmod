from pandas import Series, DataFrame
from swak.pd import ParquetReader, ColumnSelector, ParquetWriter
from swak.funcflow import apply, Fork, Pipe, Route, Sum
from swak.funcflow.loggers import PassThroughStdOut, PID_FMT
from swak.funcflow.concurrent import ProcessMap
from ..etl import to_frame
from ..config import config
from ..ml import Algo
from .log_messages import log_total_number_of_files, log_encode_file
from ..io import (
    load_tokenizer,
    discover_corpus,
    extract_file_name,
    save_config
)

LOGGER = PassThroughStdOut(__name__, config.log_level)
PID_LOGGER = PassThroughStdOut(__name__, config.log_level, PID_FMT)

read_parquet = ParquetReader()
write_parquet = ParquetWriter(config.encodings + '/{}', create=True)
select_column = ColumnSelector(config.files.column)

load_column = Pipe[[str], Series](
    PID_LOGGER.debug(log_encode_file),
    read_parquet,
    select_column
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
            to_frame
        ),
        extract_file_name
    ),
    write_parquet
)

encode_files = ProcessMap[[str], tuple[()], list](
    encode_file,
    max_workers=1,
    max_tasks_per_child=1
)

encode = Pipe[[tuple[()]], tuple[()]](
    LOGGER.debug(f'Saving config file to "{config.config_file}".'),
    save_config,
    LOGGER.info('Starting step "encode".'),
    LOGGER.debug(f'Scanning "{config.corpus}" for files.'),
    LOGGER.debug(f'Writing to "{config.encodings}".'),
    discover_corpus,
    LOGGER.debug(log_total_number_of_files),
    encode_files,
    Sum(()),
    LOGGER.info('Finished step "encode".'),
)
