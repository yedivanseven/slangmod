from swak.funcflow import Pipe, Fork
from swak.pt.types import Module
from swak.funcflow.loggers import PassThroughStdLogger
from ..config import config
from ..ml import Algo, create_generator, create_model
from ..io import (
    load_tokenizer,
    load_model,
    pre_trained_client,
    write_chat_history
)

__all__ = ['chat']

LOGGER = PassThroughStdLogger(__name__, config.log_level)


chat = Pipe[[tuple[()]], tuple[()]](
    Fork[[tuple[()]], tuple[Algo, Module]](
        LOGGER.debug(f'Loading tokenizer from "{config.tokenizer_file}".'),
        load_tokenizer,
        LOGGER.debug(f'Loading model from "{config.weights_file}".'),
        Pipe[[tuple[()]], Module](create_model, load_model)
    ),
    create_generator,
    LOGGER.info(f'Starting chat client. Enter "{config.chat.stop}" to exit.'),
    pre_trained_client,
    write_chat_history
)
