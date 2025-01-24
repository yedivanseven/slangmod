import torch as pt
from numpy import ndarray
from swak.funcflow import Pipe, Fork, Route, Map, Filter, unit
from swak.pt.create import Create
from swak.pt.types import Tensor, Module
from swak.pt.misc import Cat, LazyCatDim0
from swak.funcflow.loggers import PassThroughStdLogger
from swak.funcflow import identity
from ..config import config
from ..etl import trim_memory, Shuffle
from ..ml import (
    fold_train_sequences,
    fold_test_sequences,
    TrainData,
    TestData,
    wrap_train_data,
    wrap_test_data,
    compile_model,
    trainer,
    evaluate_model
)
from ..io import (
    save_config,
    save_model,
    discover_encodings,
    read_column,
    filter_train_files,
    filter_test_files,
    filter_validation_files
)
from .log_messages import (
    log_total_number_of_files,
    log_process_file,
    log_total_number_of_docs,
    log_remaining_number_of_sequences,
    log_number_of_tokens,
    log_data_sizes,
    log_evaluation_metrics
)

__all__ = ['train']

LOGGER = PassThroughStdLogger(__name__, config.log_level)


read_file = Pipe[[str], list[ndarray]](
    LOGGER.debug(log_process_file),
    read_column,
    list,
    LOGGER.debug(log_total_number_of_docs),
)
cat_sequences = Pipe[[list[Tensor]], Tensor](
    trim_memory,
    Cat(dim=0),
    trim_memory
)

process_train_file = Pipe[[str], Tensor](
    read_file,
LOGGER.debug(f'Dropping sequences shorter than {config.data.jitter}.'),
    Filter[list[ndarray], list](lambda seq: len(seq) > config.data.jitter),
    LOGGER.debug(log_remaining_number_of_sequences),
    trim_memory,
    Shuffle[list[ndarray]](config.data.shuffle),
    LOGGER.debug(log_number_of_tokens),
    Map[[ndarray], Tensor, list](Create(pt.long, 'cpu')),
    trim_memory,
    LOGGER.debug('Folding sequences.'),
    Map[[Tensor], Tensor, list](fold_train_sequences),
    cat_sequences
)
process_test_file = Pipe[[str], Tensor](
    read_file,
LOGGER.debug('Dropping sequences shorter than 2.'),
    Filter[list[ndarray], list](lambda seq: len(seq) > 1),
    LOGGER.debug(log_remaining_number_of_sequences),
    trim_memory,
    LOGGER.debug(log_number_of_tokens),
    Map[[ndarray], Tensor, list](Create(pt.long, 'cpu')),
    trim_memory,
    LOGGER.debug('Folding sequences.'),
    Map[[Tensor], Tensor, list](fold_test_sequences),
    cat_sequences
)

process_train = Pipe[[list[str]], TrainData](
    Map[[str], Tensor, list](process_train_file),
    Shuffle[list[Tensor]](config.data.shuffle),
    trim_memory,
    LazyCatDim0,
    trim_memory,
    wrap_train_data,
    trim_memory
)
process_test = Pipe[[list[str]], TestData](
    Map[[str], Tensor, list](process_test_file),
    trim_memory,
    Cat(dim=0),
    trim_memory,
    wrap_test_data,
    trim_memory
)

load_train = Pipe[[list[str]], TrainData](
    LOGGER.debug('Loading train data.'),
    filter_train_files,
    LOGGER.debug(log_total_number_of_files),
    process_train
)
load_test = Pipe[[list[str]], TestData](
    LOGGER.debug('Loading test data.'),
    filter_test_files,
    LOGGER.debug(log_total_number_of_files),
    process_test
)
load_validation = Pipe[[list[str]], TestData](
    LOGGER.debug('Loading validation data.'),
    filter_validation_files,
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

train_model = Pipe[[Module, TrainData, TestData], Module](
    LOGGER.info(f'Training model on {config.data.device.upper()} with a target'
                f' learning rate of {config.train.learning_rate:7.5f}'),
    trim_memory,
    trainer.train,
    LOGGER.debug(f'Saving model to "{config.model_file}".'),
    Fork[[Module], Module](
        identity,
        save_model
    )
)

train = Pipe[[tuple[()]], tuple[()]](
    LOGGER.debug(f'Saving config file to "{config.config_file}".'),
    save_config,
    LOGGER.info('Starting step "train".'),
    LOGGER.debug('Compiling model.'),
    Fork[[tuple[()]], tuple[Module, TrainData, TestData, TestData]](
        compile_model,
        load_data
    ),
    Route[[Module, TrainData, TestData, TestData], tuple[Module, TestData]](
        [(0, 1, 2), 3],
        train_model,
        identity,
    ),
    LOGGER.info('Validating model.'),
    evaluate_model,
    LOGGER.info(log_evaluation_metrics),
    LOGGER.info('Finished step "train".'),
    unit
)
