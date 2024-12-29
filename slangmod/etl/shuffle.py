import random
from collections.abc import MutableSequence
from swak.misc import ArgRepr

__all__ = ['Shuffle']


class Shuffle[T, S](ArgRepr):

    def __init__(self, active: bool = True) -> None:
        super().__init__(active)
        self.active = active

    def __call__(self, sequence: MutableSequence[S]) -> T:
        if self.active:
            random.shuffle(sequence)
        return sequence
