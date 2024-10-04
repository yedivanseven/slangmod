from ..config import config

def dry_run() -> None:
    print(repr(config))
