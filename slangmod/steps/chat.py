from swak.funcflow import Pipe, Fork, unit
from swak.pt.io import ModelLoader
from swak.pt.types import Module
from swak.funcflow.loggers import PassThroughStdOut
from ..config import config
from ..io import load_tokenizer, ConsoleClient, console_client
from ..ml import Algo, create_generator

LOGGER = PassThroughStdOut(__name__, config.log_level)

chat = Pipe[[tuple[()]], ConsoleClient](
    Fork[[tuple[()]], tuple[Algo, Module]](
        LOGGER.debug(f'Loading tokenizer from "{config.tokenizer_file}".'),
        load_tokenizer,
        LOGGER.debug(f'Loading model from "{config.model_file}".'),
        ModelLoader(config.model_file)
    ),
    create_generator,
    LOGGER.info(f'Starting chat client. Enter "{config.chat.stop}" to exit.'),
    console_client,
    unit
)
