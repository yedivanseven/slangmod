from swak.text import JsonWriter
from ..config import config

__all__ = ['write_chat_history']


write_chat_history = JsonWriter(
    config.chat_file,
    overwrite=True,
    create=True
)
