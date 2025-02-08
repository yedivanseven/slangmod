import random
from collections.abc import MutableSequence
from swak.misc import ArgRepr

__all__ = ['Shuffle']


class Shuffle[T: MutableSequence](ArgRepr):
    """Wrapper around `random.shuffle`.

    Parameters
    ----------
    active: bool, optional
        Flag to switch off shuffling for debugging purposes.
        Defaults to ``True``.

    """

    def __init__(self, active: bool = True) -> None:
        super().__init__(active)
        self.active = active

    def __call__(self, sequence: T) -> T:
        """Shuffle a mutable sequence in place.

        Parameters
        ----------
        sequence: MutableSequence
            The mutable sequence to shuffle in place.

        Returns
        -------
        MutableSequence
            The input sequence shuffled in place.

        """
        if self.active:
            random.shuffle(sequence)
        return sequence
