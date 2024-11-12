from swak.text import TomlWriter
from swak.funcflow import Partial
from ..config import config

__all__ = ['save_config']

config_writer = TomlWriter(config.config_file, True, True, True)
save_config = Partial[tuple[()]](config_writer, config)
