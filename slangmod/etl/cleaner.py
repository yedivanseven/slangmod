from typing import Any
from collections.abc import Callable
from hashlib import sha256
from tqdm import tqdm
from pandas import Series, DataFrame
from swak.misc import ArgRepr

__all__ = ['CorpusCleaner']


class CorpusCleaner(ArgRepr):

    def __init__(
            self,
            process: Callable[[str], str],
            *args: Any,
            **kwargs: Any
    ) -> None:
        super().__init__(process, *args, **kwargs)
        self.process = process
        self.args = args
        self.kwargs = kwargs

    def __call__(self, corpus: Series, **kwargs: Any) -> tuple[DataFrame, str]:
        updated = self.kwargs | kwargs
        wrapped = tqdm(corpus, *self.args, **updated)
        corpus[:] = [self.process(document) for document in wrapped]
        hashed = sha256(str(corpus).encode()).hexdigest()
        return corpus.to_frame(), hashed
