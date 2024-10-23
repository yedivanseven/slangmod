import importlib.metadata as meta
from pathlib import Path
from swak.jsonobject import JsonObject
from swak.jsonobject.fields import resolve
from .tokenizer import Tokenizer
from .data import Data
from .model import Model
from .train import Train

__all__ = ['Main']

PACKAGE = __name__.split('.')[0]
VERSION = meta.version(PACKAGE)


class Main(JsonObject):
    package: str = PACKAGE
    version: str = VERSION
    log_level: int = 10  # 10=debug, 20=info, 30=warning, 40=error, 50=critical
    corpus: resolve = '/home/georg/Projects/slangmod/data/corpus'
    workdir: resolve = '/home/georg/Projects/slangmod/data'
    tokenizer: Tokenizer = Tokenizer()
    data: Data = Data()
    model: Model = Model()
    train: Train = Train()

    @property
    def tokenizer_file(self) -> str:
        return str((Path(self.workdir) / 'tokenizer.json').resolve())

    @property
    def checkpoint_file(self) -> str:
        return str((Path(self.workdir) / 'checkpoint.pt').resolve())

    @property
    def model_file(self) -> str:
        return str((Path(self.workdir) / 'model.pt').resolve())
