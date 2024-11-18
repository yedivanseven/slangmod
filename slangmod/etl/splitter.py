import random
import math
from swak.misc import ArgRepr
from ..config import config

__all__ = ['CorpusSplitter']


class CorpusSplitter[T](ArgRepr):

    def __init__(self, test: float = 0.1, shuffle: bool = True) -> None:
        super().__init__(test, shuffle)
        self.test = test
        self.shuffle = shuffle

    @staticmethod
    def jumble(sequences: list[T]) -> list[T]:
        return random.sample(sequences, len(sequences))

    def __call__(self, sequences: list[T]) -> tuple[list[T], list[T], list[T]]:
        n_total = len(sequences)
        n_test = math.ceil(n_total * self.test)
        n_train = n_total - 2 * n_test
        jumbled = self.jumble(sequences) if self.shuffle else sequences
        return (
            jumbled[:n_train],
            jumbled[n_train:n_train + n_test],
            jumbled[n_train + n_test:]
        )


split_corpus = CorpusSplitter(config.data.test, config.data.shuffle)
