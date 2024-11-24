import torch as pt
from swak.funcflow import Pipe, Fork, Route, Map, Filter, unit
from swak.pt.create import Create
from swak.pt.types import Tensor
from swak.pt.io import ModelSaver
from swak.pt.misc import Cat
from swak.funcflow.loggers import PassThroughStdOut
from swak.funcflow import identity, apply
from ..config import config
from ..io import save_config, discover_corpus, load_tokenizer
from ..etl import split_corpus, fold_train, fold_test
from ..ml import Algo
from ..ml import TrainData, TestData
from ..ml import make_train_data, make_test_data
from ..ml import Model, compile_model
from ..ml import trainer
from ..ml import validate
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


process_train = Pipe[[list[Tensor]], Tensor](
    Map[[Tensor], Tensor, list](fold_train),
    Cat(dim=0),
    make_train_data
)
process_test = Pipe[[list[Tensor]], Tensor](
    Map[[Tensor], Tensor, list](fold_test),
    Cat(dim=0),
    make_test_data
)

load_data = Pipe[[tuple[()]], tuple[TrainData, TestData, TestData]](
    Fork[[tuple[()]], tuple[Algo, str]](
        Pipe[[tuple[()]], Algo](
            LOGGER.debug(f'Loading tokenizer "{config.tokenizer_file}".'),
            load_tokenizer
        ),
        Pipe[[tuple[()]], str](
            LOGGER.debug(f'Scanning "{config.corpus}" for files.'),
            discover_corpus,
            LOGGER.debug(log_total_number_of_files),
            load_corpus
        )
    ),
    LOGGER.debug('Encoding corpus ...'),
    apply,
    LOGGER.debug(log_total_number_of_docs),
    LOGGER.debug(f'Dropping sequences shorter than {config.data.jitter}.'),
    Filter[list[int], list](lambda seq: len(seq) > config.data.jitter),
    LOGGER.debug(log_remaining_number_of_sequences),
    LOGGER.debug(log_total_number_of_tokens),
    Map[[list[int]], Tensor, list](Create(pt.long, 'cpu')),
    LOGGER.debug('Splitting into train, test, and validation data.'),
    split_corpus,
    Route[
        [list[Tensor], list[Tensor], list[Tensor]],
        tuple[TrainData, TestData, TestData]
    ](
        [0, 1, 2],
        process_train,
        process_test,
        process_test
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
