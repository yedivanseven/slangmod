from collections.abc import Callable, Iterable
from tqdm import tqdm
from swak.misc import ArgRepr

__all__ = ['CorpusCleaner']


class CorpusCleaner(ArgRepr):

    def __init__(self, process: Callable[[str], str]) -> None:
        super().__init__(process)
        self.process = process

    def __call__(self, corpus: Iterable[str]) -> list[str]:
        wrapped = tqdm(corpus, 'Documents', leave=False)
        return [self.process(text) for text in wrapped]
