import math
import importlib.metadata as meta
from swak.jsonobject import JsonObject
from .files import Files
from .tokens import Tokens
from .data import Data
from .model import Model
from .train import Train
from .chat import Chat

__all__ = ['Main']

PACKAGE = __name__.split('.')[0]
VERSION = meta.version(PACKAGE)


class Main(JsonObject):
    package: str = PACKAGE
    version: str = VERSION
    log_level: int = 10  # 10=debug, 20=info, 30=warning, 40=error, 50=critical
    files: Files = Files()
    tokens: Tokens = Tokens()
    data: Data = Data()
    model: Model = Model()
    train: Train = Train()
    chat: Chat = Chat()

    @property
    def lr(self) -> float:
        if self.train.learning_rate is None:
            return 9 * self.train.super_batch * self.model.dim**-1.53 / 256
        return self.train.learning_rate
