from typing import Any
from collections.abc import Mapping
from swak.misc import ArgRepr
from swak.text import TomlWriter
from swak.funcflow import Pipe
from ..config import config

__all__ = ['save_config']


class NoneDropper(ArgRepr):

    def __init__(self, mapping: Mapping | None = None) -> None:
        super().__init__()
        self.mapping = mapping

    def __call__(self, mapping: Mapping | None = None) -> dict:
        return self.recursive(self.mapping if mapping is None else mapping)

    def recursive(self, mapping: Any) -> Any:
        if not isinstance(mapping, Mapping):
            return mapping
        return {
            key: self.recursive(value)
            for key, value in mapping.items()
            if value is not None
        }

drop_none = NoneDropper(config.as_json)
write_config = TomlWriter(config.config_file, True)
save_config = Pipe[[tuple[()]], tuple[()]](
    drop_none,
    write_config
)
