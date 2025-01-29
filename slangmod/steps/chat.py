from swak.funcflow import Pipe, Fork, unit
from swak.pt.io import ModelLoader
from swak.pt.types import Module
from swak.funcflow.loggers import PassThroughStdLogger
from ..config import config
from ..io import load_tokenizer, pre_trained_client
from ..ml import Algo, create_generator

__all__ = ['chat']

LOGGER = PassThroughStdLogger(__name__, config.log_level)


chat = Pipe[[tuple[()]], tuple[()]](
    Fork[[tuple[()]], tuple[Algo, Module]](
        LOGGER.debug(f'Loading tokenizer from "{config.tokenizer_file}".'),
        load_tokenizer,
        LOGGER.debug(f'Loading model from "{config.model_file}".'),
        ModelLoader(config.model_file)
    ),
    create_generator,
    LOGGER.info(f'Starting chat client. Enter "{config.chat.stop}" to exit.'),
    pre_trained_client,
    unit
)
