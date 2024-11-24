from pandas import Series
from swak.funcflow import Pipe
from swak.funcflow.loggers import PassThroughStdOut
from swak.pd import ParquetReader, ColumnSelector
from ..config import config
from ..io import save_config, save_tokenizer, discover_corpus, CorpusLoader
from ..ml import tokenizer
from .log_messages import log_total_number_of_files

LOGGER = PassThroughStdOut(__name__, config.log_level)

read_parquet = ParquetReader()
select_column = ColumnSelector('text')
loader = Pipe[[str], Series](read_parquet, select_column)
load_corpus = CorpusLoader(loader)

tokenize = Pipe[[tuple[()]], tuple[()]](
    LOGGER.debug(f'Saving config file to "{config.config_file}".'),
    save_config,
    LOGGER.info('Starting step "tokenize".'),
    LOGGER.debug(f'Scanning folder "{config.corpus}" for files.'),
    discover_corpus,
    LOGGER.debug(log_total_number_of_files),
    load_corpus,
    LOGGER.info('Training tokenizer.'),
    tokenizer.train,
    LOGGER.debug(f'Saving trained tokenizer to "{config.tokenizer_file}".'),
    save_tokenizer,
    LOGGER.info('Finished step "tokenize".')
)
