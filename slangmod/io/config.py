from swak.text import TomlWriter
from swak.funcflow import Partial
from ..config import config

__all__ = ['save_config']

config_writer = TomlWriter(
    path=config.config_file,
    overwrite=True,
    create=True,
    prune=True
)
save_config = Partial[tuple[()]](config_writer, config)
