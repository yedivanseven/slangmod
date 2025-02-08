import json
from pathlib import Path
from swak.text import TomlReader
from ..config import config

__all__ = ['summarize']

path = (Path(config.folder) / config.files.summary)
read = TomlReader(path)
files = sorted(item for item in path.glob('*.toml') if item.is_file())


def summarize() -> tuple[()]:
    summaries = [read(file) | {'start': file.stem} for file in files]
    print(json.dumps(summaries, indent=4))  # noqa: T201
    return ()
