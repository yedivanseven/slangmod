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


# ToDo: Flag to switch off progress bars
class Main(JsonObject):
    package = PACKAGE
    version = VERSION

    log_level: int = 10  # 10=debug, 20=info, 30=warning, 40=error, 50=critical
    workdir: resolve = '/home/georg/Projects/slangmod/data'
    size: Maybe[str](Lower()) = 'xxs'
    toml: Maybe[str](resolve) = None
    name: Maybe[str](Lower()) = 'xxs'
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
        if self.name is None:
            settings = ''.join([
                str(self.version),
                str(self.tokens),
                str(self.data),
                str(self.model),
                str(self.train)
            ])
            name = shake_128(settings.encode()).hexdigest(4)
        else:
            name = self.name
        path = Path(self.workdir) / name
        return str(path.resolve())

    @property
    def corpus(self) -> str:
        return str((Path(self.workdir) / 'corpus').resolve())

    @property
    def encodings(self) -> str:
        return str((Path(self.workdir) / 'encodings').resolve())

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
    def clean_files(self) -> str:
        return self.corpus + '/{}-{}.' + self.files.suffix

    @property
    def encoded_files(self) -> str:
        return self.encodings + '/{}'
