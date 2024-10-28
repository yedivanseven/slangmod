from swak.funcflow import Pipe
from swak.funcflow.loggers import PassThroughStdOut
from ..config import config
from ..io import save_tokenizer, discover_corpus
from ..ml import tokenizer
from .log_messages.tokenize import log_corpus

LOGGER = PassThroughStdOut(__name__, config.log_level)

tokenize = Pipe[[tuple[()]], tuple[()]](
    LOGGER.info('Starting step "tokenize".'),
    LOGGER.debug(f'Scanning folder "{config.files.corpus}" for books.'),
    discover_corpus,
    LOGGER.debug(log_corpus),
    LOGGER.info('Training tokenizer.'),
    tokenizer.train,
    LOGGER.debug(f'Saving trained tokenizer to "{config.files.tokenizer}".'),
    save_tokenizer,
    LOGGER.info('Finished step "tokenize".')
)
