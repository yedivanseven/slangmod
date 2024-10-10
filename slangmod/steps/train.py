import torch as pt
from swak.funcflow import Pipe, Fork, Route
from swak.pt import device
from swak.pt.create import Create
from swak.pt.types import Tensor
from swak.pt.io import ModelSaver
from swak.funcflow.loggers import PassThroughStdOut
from swak.funcflow import identity
from ..config import config
from ..io import load_books, load_tokenizer
from ..ml import Encoder
from ..ml import split_train_test_validation
from ..ml import TrainData, TestData
from ..ml import make_train_data, make_test_data, make_validation_data
from ..ml import Model, compile_model
from ..ml import trainer
from .log_messages.train import log_data_sizes

LOGGER = PassThroughStdOut(__name__, config.log_level)


load_data = Pipe[[tuple[()]], tuple[TrainData, TestData, TestData]](
    LOGGER.debug(f'Loading books from folder "{config.books}".'),
    load_books,
    LOGGER.debug(f'Loading pre-trained tokenizer "{config.tokenizer_file}".'),
    LOGGER.debug('Encoding books.'),
    Encoder(load_tokenizer()),
    Create(pt.int64, device),
    LOGGER.debug('Splitting into train, test, and validation data.'),
    split_train_test_validation,
    Route[[Tensor], tuple[TrainData, TestData, TestData]](
        [0, 1, 2],
        make_train_data,
        make_test_data,
        make_validation_data,
    ),
    LOGGER.info(log_data_sizes)
)


train_model = Pipe[[Model, TrainData, TestData], Model](
    LOGGER.info('Training model.'),
    trainer.train,
    LOGGER.debug('Saving model.'),
    Fork[[Model], Model](
        identity,
        ModelSaver(config.model_file)
    )
)

train = Pipe[[tuple[()]], tuple[Model, TrainData, TestData, TestData]](
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
    LOGGER.info('Validating model ...'),
)
