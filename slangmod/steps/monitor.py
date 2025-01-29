import importlib.resources as resources
from ..config import config

__all__ = ['monitor']

# ToDo: Interpolate config.folder into script
def monitor() -> tuple[()]:
    print(resources                                        # noqa: T201
          .files(config.package)
          .joinpath('gnuplot', 'monitor.gp')
          .read_text())
    return ()
