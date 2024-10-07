from pathlib import Path
from ..config import config

books = [
    str(item.resolve())
    for item in Path(config.books).iterdir()
    if item.is_file() and item.suffix == '.txt'
]
