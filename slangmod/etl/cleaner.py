from typing import Any
from collections.abc import Callable
from hashlib import sha256
from tqdm import tqdm
from pandas import Series, DataFrame
from swak.misc import ArgRepr

__all__ = ['CorpusCleaner']


class CorpusCleaner(ArgRepr):
    """Clean a single pandas series where each entry represents one document.

    Parameters
    ----------
    process: callable
        A callable object that accepts a single (raw) string as input and
        returns the cleaned string.
    *args
        Optional arguments to pass through to the ``tqdm`` progress bar.
    **kwargs
        Optional keyword arguments to pass through to the ``tqdm``
        progress bar.

    """

    def __init__(
            self,
            process: Callable[[str], str],
            min_len: int = 0,
            *args: Any,
            **kwargs: Any
    ) -> None:
        super().__init__(process, min_len, *args, **kwargs)
        self.process = process
        self.min_len = min_len
        self.args = args
        self.kwargs = kwargs

    def __call__(self, corpus: Series, **kwargs: Any) -> tuple[DataFrame, str]:
        """Apply the cached processor to each document in a corpus.

        Parameters
        ----------
        corpus: Series
            Pandas Series with each entry representing a single document to
            clean, that is, a single string.
        **kwargs
            Optional keyword arguments are merged into the keyword arguments
            given at instantiation and passed on to the ``tqdm`` progress bar.

        Returns
        -------
        DataFrame
            A pandas dataframe with the cleaned series as a sole column.
        str
            The SHA256 hash of the cleaned series for downstream deduplication.

        """
        merged_kwargs = self.kwargs | kwargs
        wrapped = tqdm(corpus, *self.args, **merged_kwargs)
        processed = (self.process(document) for document in wrapped)
        filtered = filter(lambda doc: len(doc) >= self.min_len, processed)
        corpus = Series(filtered, name=corpus.name)
        #corpus[:] = [self.process(document) for document in wrapped]
        hashed = sha256(str(corpus).encode()).hexdigest()
        return corpus.to_frame(), hashed
