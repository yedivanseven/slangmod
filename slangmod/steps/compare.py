import importlib.resources as resources


def compare() -> tuple[()]:
    print(resources                                        # noqa: T201
          .files('slangmod')
          .joinpath('gnuplot', 'compare.gp')
          .read_text())
    return ()
