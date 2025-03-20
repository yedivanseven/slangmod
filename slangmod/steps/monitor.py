import importlib.resources as resources
from ..config import config

__all__ = ['monitor']


def monitor() -> tuple[()]:
    template = resources.files(
        config.package
    ).joinpath(
        'gnuplot', 'monitor.gp'
    ).read_text()
    script = template.format(
        folder=config.folder,
        subdir=config.files.monitor,
        clip=config.train.clip_grad
    )
    print(script) # noqa: T201
    return ()
