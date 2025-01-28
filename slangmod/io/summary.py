from typing import Any
from collections.abc import Callable
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

write_toml = TomlWriter(
    path=config.summary_file,
    overwrite=True,
    create=True,
    prune=True
)


class TrainTomlPrinter(TrainPrinter):
    """Assemble a TOML-formatted string with a summary of the training run.

    Parameters
    ----------
    printer: callable, optional
        Will be called with the assembled message. Defaults to the python
        builtin ``print`` function, but could also be a logging command.

    """

    # ToDo: Rethink this. Maybe better dict of lists, not list of dicts.
    TEMPLATE = ('[[training.epochs]]\n'
                'train_loss = {:7.5f}\n'
                'test_loss = {:7.5f}\n'
                'learning_rate = {:7.5f}\n')

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
        epochs = '\n'.join([
            self.TEMPLATE.format(train_loss, test_loss, lr)
            for train_loss, test_loss, lr
            in zip(history['train_loss'], history['test_loss'], history['lr'])
        ])
        msg = ( '\n[training]\n'
               f'last_epoch = {epoch - 1}\n'
               f'best_epoch = {best_epoch - 1}\n'
               f'best_loss = {best_loss:7.5f}\n'
               f'max_epochs_reached = {str(max_epochs_reached).lower()}\n\n'
               f'{epochs}')
        self.printer(msg)


# Provide ready-to-use instances of config saver and TrainTomlPrinter
save_config = Partial[tuple[()]](write_toml, config)
save_train_toml = TrainTomlPrinter(SUMMARY.info)
