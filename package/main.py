from swak.cli import Importer
from .config import config, actions

# Import the steps to execute based on what was parsed from the command line
import_steps = Importer(config.package)
steps = import_steps(*actions)


def run() -> None:
    for step in steps:
        step()


if __name__ == '__main__':
    run()
