from ..config import config

__all__ = ['dry_run']


def dry_run() -> None:
    print(repr(config))  # noqa: T201
