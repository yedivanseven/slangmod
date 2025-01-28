import importlib.metadata as meta
import datetime as dt
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
    package = PACKAGE
    version = VERSION
    start = dt.datetime.now().strftime("%Y-%m-%d_%Hh%Mm%Ss")

    log_level: int = 10  # 10=debug, 20=info, 30=warning, 40=error, 50=critical
    workdir: resolve = '/home/georg/Projects/slangmod/data'
    size: Maybe[str](Lower()) = 'dev'
    toml: Maybe[str](resolve) = None
    name: Maybe[str](Lower()) = 'dev'
    progress: bool = True
    actions: list = []
    files: Files = Files()
    tokens: Tokens = Tokens()
    data: Data = Data()
    model: Model = Model()
    train: Train = Train()
    chat: Chat = Chat()

    @property
    def resume(self) -> bool:
        return 'resume' in self.actions

    @property
    def mode(self) -> str:
        return 'a' if self.resume else 'w'

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
        return str((Path(self.workdir) / self.files.tokenizer).resolve())

    @property
    def checkpoint_file(self) -> str:
        return str((Path(self.folder) / self.files.checkpoint).resolve())

    @property
    def model_file(self) -> str:
        return str((Path(self.folder) / self.files.model).resolve())

    @property
    def summary_file(self) -> str:
        return self.add_time(self.files.summary)

    @property
    def log_file(self) -> str:
        return str((Path(self.folder) / self.files.log).resolve())

    @property
    def monitor_file(self):
        return self.add_time(self.files.monitor)

    @property
    def clean_files(self) -> str:
        return self.corpus + '/{}-{}.' + self.files.suffix

    @property
    def encoded_files(self) -> str:
        return self.encodings + '/{}'

    def add_time(self, file: str) -> str:
        """Add datetime suffix to files if we're not resuming."""
        *parts, extension = file.split('.')
        stem = '.'.join(parts)
        files = sorted(Path(self.folder).glob(f'{stem}*.{extension}'))
        if files and self.resume:
            return str(files[-1].resolve())
        new = stem + f'_{self.start}.' + extension
        return str((Path(self.folder) / new).resolve())
