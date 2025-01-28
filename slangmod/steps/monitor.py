import importlib.resources as resources


def monitor() -> tuple[()]:
    print(resources                                        # noqa: T201
          .files('slangmod')
          .joinpath('gnuplot', 'monitor.gp')
          .read_text())
    return ()
