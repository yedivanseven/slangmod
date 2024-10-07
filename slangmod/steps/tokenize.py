from tokenizers import Tokenizer
from swak.funcflow import Pipe, Partial
from swak.funcflow.loggers import PassThroughStdOut
from ..config import config
from ..io import save_tokenizer, load_books
from ..ml import train_tokenizer

LOGGER = PassThroughStdOut(__name__, config.log_level)

tokenize = Pipe[tuple[()], [tuple[()]]](
    LOGGER.debug(f'Scanning folder "{config.books}" for books.'),
    LOGGER.debug(f'Found books:\n{load_books}\nlogging from module'),
    LOGGER.info('Training tokenizer.'),
    Partial[Tokenizer](
        train_tokenizer,
        load_books.books
    ),
    LOGGER.info(f'Saving trained tokenizer to "{config.tokenizer_file}"'),
    save_tokenizer,
    LOGGER.info('Done training tokenizer.')
)
