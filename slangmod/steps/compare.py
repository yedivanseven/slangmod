import importlib.resources as resources
from ..config import config

__all__ = ['compare']


def compare() -> tuple[()]:
    template = (
        resources
        .files(config.package)
        .joinpath('gnuplot', 'compare.gp')
        .read_text()
    )
    script = template.format(folder=config.folder, subdir=config.files.monitor)
    print(script)  # noqa: T201
    return ()
