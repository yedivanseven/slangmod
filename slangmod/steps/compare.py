import importlib.resources as resources
from ..config import config

__all__ = ['compare']


# ToDo: Interpolate config.folder into script and make persistent
def compare() -> tuple[()]:
    print(resources                                        # noqa: T201
          .files(config.package)
          .joinpath('gnuplot', 'compare.gp')
          .read_text())
    return ()
