from typing import Any, overload
from pandas import Series, DataFrame
from swak.misc import ArgRepr

__all__ = [
    'DuplicateDropper',
    'drop_duplicates'
]


class DuplicateDropper(ArgRepr):

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs


    @overload
    def __call__(self, pd: Series, **kwargs: Any) -> Series:
        ...

    @overload
    def __call__(self, pd: DataFrame, **kwargs: Any) -> DataFrame:
        ...

    def __call__(self, pd, **kwargs):
        merged_kwargs = self.kwargs | kwargs
        return pd.drop_duplicates(*self.args, **merged_kwargs)


drop_duplicates = DuplicateDropper(inplace=True, ignore_index=True)
