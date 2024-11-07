from swak.dictionary import NoneDropper
from swak.text import TomlWriter
from swak.funcflow import Pipe
from ..config import config

__all__ = ['save_config']

drop_none = NoneDropper(config.as_json)
write_config = TomlWriter(config.config_file, True)
save_config = Pipe[[tuple[()]], tuple[()]](
    drop_none,
    write_config
)
