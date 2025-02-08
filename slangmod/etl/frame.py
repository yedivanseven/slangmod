from typing import Any
from collections.abc import Hashable, Iterable
from pandas import Series, DataFrame
from swak.misc import ArgRepr
from ..config import config


class ToFrame(ArgRepr):
    """Turn any iterable into a single-column pandas dataframe.

    Parameters
    ----------
    name: Hashable
        The name of the single column in the dataframe.
    **kwargs
        Additional keyword arguments are forwarded to the pandas ``Series``
        `constructor <https://pandas.pydata.org/pandas-docs/stable/
        reference/api/pandas.Series.html>`_.

    """

    def __init__(self, name: Hashable, **kwargs: Any) -> None:
        self.name = name
        self.kwargs = kwargs
        super().__init__(name, **kwargs)

    def __call__(self, iterable: Iterable) -> DataFrame:
        """Convert an iterable into a single-column pandas dataframe.

        Parameters
        ----------
        iterable: iterable
            The object to convert into a single-column pandas dataframe.

        Returns
        -------
        DataFrame
            A pandas dataframe with the contents of the `iterable` in its
            only column.

        """
        return Series(iterable, name=self.name, **self.kwargs).to_frame()


# Provide a ready-to use instance of ToFrame
to_frame = ToFrame(name=config.files.column)
