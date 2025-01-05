from swak.funcflow import Pipe
from swak.funcflow.loggers import PassThroughStdOut
from ..config import config
from ..ml import tokenizer
from ..etl import trim_memory
from ..io import (
    save_config,
    save_tokenizer,
    discover_corpus,
    load_corpus
)
from .log_messages import log_total_number_of_files

LOGGER = PassThroughStdOut(__name__, config.log_level)

tokenize = Pipe[[tuple[()]], tuple[()]](
    LOGGER.debug(f'Saving config file to "{config.config_file}".'),
    save_config,
    LOGGER.info('Starting step "tokenize".'),
    LOGGER.debug(f'Scanning folder "{config.corpus}" for files.'),
    discover_corpus,
    LOGGER.debug(log_total_number_of_files),
    load_corpus,
    LOGGER.info(f'Training tokenizer {config.tokens.algo}.'),
    tokenizer.train,
    LOGGER.debug(f'Saving trained tokenizer to "{config.tokenizer_file}".'),
    save_tokenizer,
    trim_memory,
    LOGGER.info('Finished step "tokenize".')
)
