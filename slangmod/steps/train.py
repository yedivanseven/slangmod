import torch as pt
from swak.funcflow import Pipe, Route
from swak.pt import device
from swak.pt.create import Create
from swak.pt.types import Tensor
from swak.funcflow.loggers import PassThroughStdOut
from ..config import config
from ..io import load_books, load_tokenizer
from ..ml import Encoder
from ..ml import split_train_test_validation
from ..ml import TrainData, TestData
from ..ml import make_train_data, make_test_data, make_validation_data
from .log_messages.train import log_data_sizes

LOGGER = PassThroughStdOut(__name__, config.log_level)


train = Pipe[tuple[()], [TrainData, TestData, TestData]](
    LOGGER.info('Starting step "train".'),
    LOGGER.debug(f'Loading books from folder "{config.books}".'),
    load_books,
    LOGGER.debug('Loading pre-trained tokenizer and encoding books.'),
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
