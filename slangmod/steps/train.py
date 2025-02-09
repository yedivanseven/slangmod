import torch as pt
from numpy import ndarray
from swak.funcflow import Pipe, Fork, Route, Map, Filter, unit
from swak.pt.create import Create
from swak.pt.types import Tensor, Module
from swak.pt.misc import Cat, LazyCatDim0
from swak.funcflow.loggers import (
    PassThroughStdLogger,
    PassThroughFileLogger,
    SHORT_FMT,
    RAW_FMT
)
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
    trainer,
    create_model,
    compile_model,
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
    log_evaluation_metrics,
    save_evaluation_metrics
)

__all__ = ['train']

LOG_TERM = PassThroughStdLogger(__name__, config.log_level)
LOG_FILE = PassThroughFileLogger(
    config.log_file,
    fmt=SHORT_FMT,
    mode=config.mode
)
SUMMARY = PassThroughFileLogger(config.summary_file, fmt=RAW_FMT, mode='a')

read_file = Pipe[[str], list[ndarray]](
    LOG_TERM.debug(log_process_file),
    LOG_FILE.debug(log_process_file),
    read_column,
    list,
    LOG_TERM.debug(log_total_number_of_docs),
    LOG_FILE.debug(log_total_number_of_docs),
)
cat_sequences = Pipe[[list[Tensor]], Tensor](
    trim_memory,
    Cat(dim=0),
    trim_memory
)

process_train_file = Pipe[[str], Tensor](
    read_file,
LOG_TERM.debug(f'Dropping sequences shorter than {config.data.jitter}.'),
    LOG_FILE.debug(f'Dropping sequences shorter than {config.data.jitter}.'),
    Filter[list[ndarray], list](lambda seq: len(seq) > config.data.jitter),
    LOG_TERM.debug(log_remaining_number_of_sequences),
    LOG_FILE.debug(log_remaining_number_of_sequences),
    trim_memory,
    Shuffle[list[ndarray]](config.data.shuffle),
    LOG_TERM.debug(log_number_of_tokens),
    LOG_FILE.debug(log_number_of_tokens),
    Map[[ndarray], Tensor, list](Create(pt.long, 'cpu')),
    trim_memory,
    LOG_TERM.debug('Folding sequences.'),
    LOG_FILE.debug('Folding sequences.'),
    Map[[Tensor], Tensor, list](fold_train_sequences),
    cat_sequences
)
process_test_file = Pipe[[str], Tensor](
    read_file,
LOG_TERM.debug('Dropping sequences shorter than 2.'),
    LOG_FILE.debug('Dropping sequences shorter than 2.'),
    Filter[list[ndarray], list](lambda seq: len(seq) > 1),
    LOG_TERM.debug(log_remaining_number_of_sequences),
    LOG_FILE.debug(log_remaining_number_of_sequences),
    trim_memory,
    LOG_TERM.debug(log_number_of_tokens),
    LOG_FILE.debug(log_number_of_tokens),
    Map[[ndarray], Tensor, list](Create(pt.long, 'cpu')),
    trim_memory,
    LOG_TERM.debug('Folding sequences.'),
    LOG_FILE.debug('Folding sequences.'),
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
    LOG_TERM.debug('Loading train data.'),
    LOG_FILE.debug('Loading train data.'),
    filter_train_files,
    LOG_TERM.debug(log_total_number_of_files),
    LOG_FILE.debug(log_total_number_of_files),
    process_train
)
load_test = Pipe[[list[str]], TestData](
    LOG_TERM.debug('Loading test data.'),
    LOG_FILE.debug('Loading test data.'),
    filter_test_files,
    LOG_TERM.debug(log_total_number_of_files),
    LOG_FILE.debug(log_total_number_of_files),
    process_test
)
load_validation = Pipe[[list[str]], TestData](
    LOG_TERM.debug('Loading validation data.'),
    LOG_FILE.debug('Loading validation data.'),
    filter_validation_files,
    LOG_TERM.debug(log_total_number_of_files),
    LOG_FILE.debug(log_total_number_of_files),
    process_test
)

load_data = Pipe[[tuple[()]], tuple[TrainData, TestData, TestData]](
    LOG_TERM.debug(f'Scanning "{config.encodings}" for files.'),
    LOG_FILE.debug(f'Scanning "{config.encodings}" for files.'),
    discover_encodings,
    LOG_TERM.debug(log_total_number_of_files),
    LOG_FILE.debug(log_total_number_of_files),
    Fork[[list[str]], tuple[TrainData, TestData, TestData]](
        load_train,
        load_test,
        load_validation
    ),
    LOG_TERM.info(log_data_sizes),
    LOG_FILE.info(log_data_sizes)
)

prepare_model = Pipe[[tuple[()]], Module](
    LOG_TERM.debug('Instantiating model.'),
    LOG_FILE.debug('Instantiating model.'),
    create_model,
    LOG_TERM.debug('Compiling model.'),
    LOG_FILE.debug('Compiling model.'),
    compile_model
)

train_model = Pipe[[Module, TrainData, TestData], Module](
    LOG_TERM.info(f'Training model on {config.data.device.upper()} '
                  f'with a maximum learning rate of '
                  f'{config.train.learning_rate:7.5f}'),
   LOG_FILE.info(f'Training model on {config.data.device.upper()} '
                  f'with a maximum learning rate of '
                  f'{config.train.learning_rate:7.5f}'),
    trim_memory,
    trainer.resume if config.resume else trainer.train,
    LOG_TERM.debug(f'Saving model to "{config.model_file}".'),
    LOG_TERM.debug(f'Saving model weights to "{config.weights_file}".'),
    LOG_FILE.debug(f'Saving model to "{config.model_file}".'),
    LOG_FILE.debug(f'Saving model weights to "{config.weights_file}".'),
    Fork[[Module], Module](
        identity,
        save_model
    )
)

train = Pipe[[tuple[()]], tuple[()]](
    LOG_TERM.debug(f'Saving config to "{config.summary_file}".'),
    LOG_FILE.debug(f'Saving config to "{config.summary_file}".'),
    save_config,
    LOG_TERM.info(f'{"Resum" if config.resume else "Start"}ing step "train".'),
    LOG_FILE.info(f'{"Resum" if config.resume else "Start"}ing step "train".'),
    Fork[[tuple[()]], tuple[Module, TrainData, TestData, TestData]](
        prepare_model,
        load_data
    ),
    Route[[Module, TrainData, TestData, TestData], tuple[Module, TestData]](
        [(0, 1, 2), 3],
        train_model,
        identity,
    ),
    LOG_TERM.info('Validating model.'),
    LOG_FILE.info('Validating model.'),
    evaluate_model,
    LOG_TERM.info(log_evaluation_metrics),
    LOG_FILE.info(log_evaluation_metrics),
    SUMMARY.info(save_evaluation_metrics),
    LOG_TERM.info('Finished step "train".'),
    LOG_FILE.info('Finished step "train".'),
    unit
)
