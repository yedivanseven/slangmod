import torch as pt
from swak.funcflow import Pipe, Fork, Route, unit
from swak.pt.create import Create
from swak.pt.types import Tensor
from swak.pt.io import ModelSaver
from swak.funcflow.loggers import PassThroughStdOut
from swak.funcflow import identity, apply
from ..config import config
from ..io import save_config, discover_corpus, load_corpus, load_tokenizer
from ..ml import Algo
from ..ml import split_data
from ..ml import TrainData, TestData
from ..ml import make_train_data, make_test_data, make_validation_data
from ..ml import Model, compile_model
from ..ml import trainer
from ..ml import validate
from .log_messages.train import log_data_sizes, log_validation_metrics

LOGGER = PassThroughStdOut(__name__, config.log_level)

load_data = Pipe[[tuple[()]], tuple[TrainData, TestData, TestData]](
    Fork[[tuple[()]], tuple[Algo, str]](
        Pipe[[tuple[()]], Algo](
            LOGGER.debug(f'Loading tokenizer "{config.tokenizer_file}".'),
            load_tokenizer
        ),
        Pipe[[tuple[()]], str](
            LOGGER.debug(f'Scanning "{config.corpus}" for files.'),
            discover_corpus,
            LOGGER.debug(f'Loading files from "{config.corpus}".'),
            load_corpus
        )
    ),
    LOGGER.debug('Encoding corpus.'),
    apply,
    LOGGER.debug('Converting to CPU tensor.'),
    Create(pt.int64, 'cpu'),
    LOGGER.debug('Splitting into train, test, and validation data.'),
    split_data,
    Route[[Tensor], tuple[TrainData, TestData, TestData]](
        [0, 1, 2],
        make_train_data,
        make_test_data,
        make_validation_data,
    ),
    LOGGER.info(log_data_sizes)
)

train_model = Pipe[[Model, TrainData, TestData], Model](
    LOGGER.info(f'Training model on {config.data.device.upper()}'
                f' with a target learning rate of {config.lr:7.5f}'),
    trainer.train,
    LOGGER.debug(f'Saving model to "{config.model_file}".'),
    Fork[[Model], Model](
        identity,
        ModelSaver(config.model_file)
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
