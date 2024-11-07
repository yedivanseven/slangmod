import importlib.metadata as meta
from hashlib import shake_128
from pathlib import Path
from swak.jsonobject import JsonObject
from swak.jsonobject.fields import Lower, resolve, Maybe
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
    corpus: resolve = '/home/georg/Projects/slangmod/data/corpus'
    workdir: resolve = '/home/georg/Projects/slangmod/data'
    size: Maybe[str](Lower()) = None
    toml: Maybe[str](resolve) = None
    name: Maybe[str](Lower()) = None
    files: Files = Files()
    tokens: Tokens = Tokens()
    data: Data = Data()
    model: Model = Model()
    train: Train = Train()
    chat: Chat = Chat()

    @property
    def preset(self) -> str:
        if self.size is None:
            return 'none.toml'
        return self.size.strip() + '.toml'

    @property
    def folder(self) -> str:
        settings = (
            str(self.version) +
            str(self.tokens) +
            str(self.data) +
            str(self.model) +
            str(self.train)
        )
        suffix = shake_128(settings.encode()).hexdigest(4)
        name = self.name + '-' + suffix if self.name else suffix
        path = Path(self.workdir) / name
        path.mkdir(parents=False, exist_ok=True)
        return str(path.resolve())

    @property
    def tokenizer_file(self) -> str:
        return str((Path(self.folder) / self.files.tokenizer).resolve())

    @property
    def checkpoint_file(self) -> str:
        return str((Path(self.folder) / self.files.checkpoint).resolve())

    @property
    def model_file(self) -> str:
        return str((Path(self.folder) / self.files.model).resolve())

    @property
    def config_file(self) -> str:
        return str((Path(self.folder) / self.files.config).resolve())

    @property
    def lr(self) -> float:
        if self.train.learning_rate is None:
            return 9 * self.train.super_batch * self.model.dim**-1.53 / 256
        return self.train.learning_rate
