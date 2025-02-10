from collections.abc import Callable
from hashlib import sha256
from tqdm import tqdm
from pandas import Series, DataFrame
from swak.misc import ArgRepr

__all__ = ['CorpusCleaner']


# ToDo: Clean files in parallel
class CorpusCleaner(ArgRepr):
    """Clean a single pandas series where each entry represents one document.

    Parameters
    ----------
    process: callable
        A callable object that accepts a single (raw) string as input and
        returns the cleaned string.
    min_len: int, optional
        The minimum number of characters that a document should have. Shorter
        documents are filtered out. Defaults to 1.
    show_progress: bool, optional
        Whether to show a progress bar that provides visual feedback in the
        console during the cleaning process. Defaults to ``True``.

    """

    def __init__(
            self,
            process: Callable[[str], str],
            min_len: int = 1,
            show_progress: bool = True
    ) -> None:
        super().__init__(process, min_len, show_progress)
        self.process = process
        self.min_len = min_len
        self.show_progress = show_progress

    def __call__(self, corpus: Series) -> tuple[DataFrame, str]:
        """Apply the cached processor to each document in a corpus.

        Parameters
        ----------
        corpus: Series
            Pandas series with each entry representing a single document to
            clean, that is, a single string.

        Returns
        -------
        DataFrame
            A pandas dataframe with the cleaned series as a sole column.
        str
            The SHA256 hash of the cleaned series.

        """
        wrapped = tqdm(corpus, 'Documents', disable=not self.show_progress)
        processed = (self.process(document) for document in wrapped)
        filtered = filter(lambda doc: len(doc) >= self.min_len, processed)
        corpus = Series(filtered, name=corpus.name)
        hashed = sha256(str(corpus).encode()).hexdigest()
        return corpus.to_frame(), hashed
