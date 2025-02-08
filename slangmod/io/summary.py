from typing import Any
from collections.abc import Callable
import tomli_w
from swak.text import TomlWriter
from swak.funcflow import Partial
from swak.pt.train import TrainPrinter, History
from swak.misc import FileLogger, RAW_FMT
from ..config import config

__all__ = [
    'save_config',
    'TrainTomlPrinter',
    'save_train_toml'
]

SUMMARY = FileLogger(config.summary_file, fmt=RAW_FMT, mode='a')


class TrainTomlPrinter(TrainPrinter):
    """Assemble a TOML-formatted string with a summary of the training run.

    Parameters
    ----------
    printer: callable, optional
        Will be called with the assembled message. Defaults to the python
        builtin ``print`` function, but could also be a logging command.

    """

    def __init__(self, printer: Callable[[str], Any] = print) -> None:
        super().__init__(printer)

    def __call__(
            self,
            epoch: int,
            best_epoch: int,
            best_loss: float,
            max_epochs_reached: bool,
            history: History
    ) -> None:
        """Assemble a summary of model training and call the printer with it.

        Parameters
        ----------
        epoch: int
            The last epoch in the training loop.
        best_epoch: int
            The epoch with the lowest loss encountered.
        best_loss: float
            The lowest loss encountered.
        max_epochs_reached: bool
            Whether the maximum number of epochs was exhausted or not.
        history: History
            Dictionary with lists of train losses, test losses, and learning
            rates.

        """
        convergence = {
            'last_epoch': epoch - 1,
            'best_epoch': best_epoch - 1,
            'best_loss': best_loss,
            'max_epochs_reached': max_epochs_reached,
            'train_loss': history['train_loss'],
            'test_loss': history['test_loss'],
            'learning_rate': history['lr']
        }
        msg = tomli_w.dumps({'convergence': convergence}, indent=4)
        self.printer('\n' + msg)


# Provide ready-to-use instances of config saver and TrainTomlPrinter
write_toml = TomlWriter(
    path=config.summary_file,
    overwrite=True,
    create=True,
    prune=True
)
save_config = Partial[tuple[()]](write_toml, config)
save_train_toml = TrainTomlPrinter(SUMMARY.info)
