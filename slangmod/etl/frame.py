from pandas import Series, DataFrame
from swak.misc import ArgRepr


class ToFrame(ArgRepr):

    def __init__(self, col: str) -> None:
        self.col = col.strip()
        super().__init__(self.col)

    def __call__(self, encodings: list[list[int]]) -> DataFrame:
        return Series(encodings, name=self.col).to_frame()


to_frame = ToFrame('text')
