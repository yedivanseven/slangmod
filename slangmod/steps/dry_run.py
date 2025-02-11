from ..config import config

__all__ = ['dry_run']

TITLE = f'{config.package} v{config.version} at {config.start}'


def dry_run() -> None:
    print(f'{TITLE}\n\n{config!r}')  # noqa: T201
