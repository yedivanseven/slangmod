import torch as pt
from swak.funcflow import Pipe, Fork, Route, Map, Filter, unit
from swak.pd import ParquetReader, ColumnSelector
from swak.pt.create import Create
from swak.pt.types import Tensor
from swak.pt.io import ModelSaver
from swak.pt.misc import Cat
from swak.funcflow.loggers import PassThroughStdOut
from swak.funcflow import identity
from ..config import config
from ..etl import fold_train, fold_test, trim_memory
from ..ml import TrainData, TestData
from ..ml import make_train_data, make_test_data
from ..ml import Model, compile_model
from ..ml import trainer
from ..ml import validate
from ..io import (
    save_config,
    discover_encodings,
    train_filter,
    test_filter,
    validation_filter
)
from .log_messages import (
    log_total_number_of_files,
    log_process_file,
    log_total_number_of_docs,
    log_remaining_number_of_sequences,
    log_total_number_of_tokens,
    log_data_sizes,
    log_validation_metrics
)

LOGGER = PassThroughStdOut(__name__, config.log_level)

read_parquet = ParquetReader()
select_column = ColumnSelector(config.files.column)

filter_train = Filter[str, list](train_filter)
filter_test = Filter[str, list](test_filter)
filter_validation = Filter[str, list](validation_filter)

read_file = Pipe[[str], list[Tensor]](
    LOGGER.debug(log_process_file),
    read_parquet,
    select_column,
    LOGGER.debug(log_total_number_of_docs),
    LOGGER.debug(log_total_number_of_tokens),
    Map(Create(pt.long, 'cpu'), list),
    trim_memory
)
cat_sequences = Pipe[[list[Tensor]], Tensor](
    trim_memory,
    Cat(dim=0),
    trim_memory
)

process_train_file = Pipe[[str], Tensor](
    read_file,
LOGGER.debug(f'Dropping sequences shorter than {config.data.jitter}.'),
    Filter[list[Tensor], list](lambda seq: len(seq) > config.data.jitter),
    LOGGER.debug(log_remaining_number_of_sequences),
    trim_memory,
    Map(fold_train),
    cat_sequences
)
process_test_file = Pipe[[str], Tensor](
    read_file,
    Map(fold_test),
    cat_sequences
)

process_train = Pipe[[list[str]], TrainData](
    Map(process_train_file),
    trim_memory,
    Cat(dim=0),
    trim_memory,
    make_train_data,
    trim_memory
)
process_test = Pipe[[list[str]], TestData](
    Map(process_test_file),
    trim_memory,
    Cat(dim=0),
    trim_memory,
    make_test_data,
    trim_memory
)

load_train = Pipe[[list[str]], TrainData](
    LOGGER.debug('Loading train data.'),
    filter_train,
    LOGGER.debug(log_total_number_of_files),
    process_train
)
load_test = Pipe[[list[str]], TestData](
    LOGGER.debug('Loading test data.'),
    filter_test,
    LOGGER.debug(log_total_number_of_files),
    process_test
)
load_validation = Pipe[[list[str]], TestData](
    LOGGER.debug('Loading validation data.'),
    filter_validation,
    LOGGER.debug(log_total_number_of_files),
    process_test
)

load_data = Pipe[[tuple[()]], tuple[TrainData, TestData, TestData]](
    LOGGER.debug(f'Scanning "{config.encodings}" for files.'),
    discover_encodings,
    LOGGER.debug(log_total_number_of_files),
    Fork[[list[str]], tuple[TrainData, TestData, TestData]](
        load_train,
        load_test,
        load_validation
    ),
    LOGGER.info(log_data_sizes)
)

train_model = Pipe[[Model, TrainData, TestData], Model](
    LOGGER.info(f'Training model on {config.data.device.upper()} with a target'
                f' learning rate of {config.train.learning_rate:7.5f}'),
    trainer.train,
    LOGGER.debug(f'Saving model to "{config.model_file}".'),
    Fork[[Model], Model](
        identity,
        ModelSaver(config.model_file, True)
    )
)

train = Pipe[[tuple[()]], tuple[Model, TrainData, TestData, TestData]](
    LOGGER.debug(f'Saving config file to "{config.config_file}".'),
    save_config,
    LOGGER.info('Starting step "train".'),
    LOGGER.debug('Compiling model.'),
    Fork[[tuple[()]], tuple[Model, TrainData, TestData, TestData]](
        compile_model,
        load_data
    ),
    Route[[Model, TrainData, TestData, TestData], tuple[Model, TestData]](
        [(0, 1, 2), 3],
        train_model,
        identity,
    ),
    LOGGER.info('Validating model.'),
    validate,
    LOGGER.info(log_validation_metrics),
    LOGGER.info('Finished step "train".'),
    unit
)
