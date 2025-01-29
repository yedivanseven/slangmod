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
    start = dt.datetime.now().strftime("%Y-%m-%d.%Hh%Mm%Ss")

    log_level: int = 10  # 10=debug, 20=info, 30=warning, 40=error, 50=critical
    workdir: resolve = '/home/georg/Projects/slangmod/data'
    size: Maybe[str](Lower()) = '1x1_32'
    toml: Maybe[str](resolve) = None
    name: Maybe[str](Lower()) = '1x1_32'
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
        return self.add_time(self.files.summary, 'toml')

    @property
    def log_file(self) -> str:
        return str((Path(self.folder) / self.files.log).resolve())

    @property
    def monitor_file(self):
        return self.add_time(self.files.monitor, 'txt')

    @property
    def clean_files(self) -> str:
        return self.corpus + '/{}-{}.' + self.files.suffix

    @property
    def encoded_files(self) -> str:
        return self.encodings + '/{}'

    def add_time(self, subdir: str, suffix: str) -> str:
        """Generate new datetime file names if we're not resuming."""
        folder = Path(self.folder)
        files = sorted((folder / subdir).glob(f'*.{suffix}'))
        if files and self.resume:
            return str(files[-1].resolve())
        new = f'{self.start}.{suffix}'
        return str((folder / subdir / new).resolve())
