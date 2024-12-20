import torch as pt
from numpy import ndarray
from swak.funcflow import Pipe, Fork, Route, Map, Filter, unit
from swak.pt.create import Create
from swak.pt.types import Tensor
from swak.pt.io import ModelSaver
from swak.pt.misc import Cat
from swak.funcflow.loggers import PassThroughStdOut
from swak.funcflow import identity
from ..config import config
from ..etl import fold_train, fold_test
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
from .tokenize import load_corpus
from .log_messages import (
    log_total_number_of_files,
    log_total_number_of_docs,
    log_remaining_number_of_sequences,
    log_total_number_of_tokens,
    log_data_sizes,
    log_validation_metrics
)

LOGGER = PassThroughStdOut(__name__, config.log_level)

filter_train = Filter[str, list](train_filter)
filter_test = Filter[str, list](test_filter)
filter_validation = Filter[str, list](validation_filter)

process_train = Pipe[[list[str]], TrainData](
    load_corpus,
    list,
    LOGGER.debug(log_total_number_of_docs),
    LOGGER.debug(f'Dropping sequences shorter than {config.data.jitter}.'),
    Filter[list[ndarray], list](lambda seq: len(seq) > config.data.jitter),
    LOGGER.debug(log_remaining_number_of_sequences),
    LOGGER.debug(log_total_number_of_tokens),
    Map[[list[int]], Tensor, list](Create(pt.long, 'cpu')),
    Map[[Tensor], Tensor, list](fold_train),
    Cat(dim=0),
    make_train_data
)
process_test = Pipe(
    load_corpus,
    list,
    LOGGER.debug(log_total_number_of_docs),
    LOGGER.debug(log_total_number_of_tokens),
    Map[[list[int]], Tensor, list](Create(pt.long, 'cpu')),
    Map[[Tensor], Tensor, list](fold_test),
    Cat(dim=0),
    make_test_data
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
    process_test
)
load_validation = Pipe[[list[str]], TestData](
    LOGGER.debug('Loading validation data.'),
    filter_validation,
    process_test
)

load_data = Pipe[[tuple[()]], tuple[TrainData, TestData, TestData]](
    LOGGER.debug(f'Scanning "{config.corpus}" for files.'),
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
