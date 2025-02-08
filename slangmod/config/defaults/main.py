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
    default = resolve(f'{PACKAGE}.toml')

    log_level: int = 10  # 10=debug, 20=info, 30=warning, 40=error, 50=critical
    workdir: resolve = '.'
    toml: Maybe[str](resolve) = None
    size: Maybe[str](Lower()) = None
    name: Maybe[str](Lower()) = None
    progress: bool = True
    _actions: list = []
    files: Files = Files()
    tokens: Tokens = Tokens()
    data: Data = Data()
    model: Model = Model()
    train: Train = Train()
    chat: Chat = Chat()

    @property
    def resume(self) -> bool:
        """Whether we are resuming certain steps or re-running from scratch."""
        return 'resume' in self._actions

    @property
    def mode(self) -> str:
        """Append log files if we resume."""
        return 'a' if self.resume else 'w'

    @property
    def preset(self) -> str:
        """If no size preset is invoked, we simply read an empty toml."""
        if self.size is None:
            return 'none.toml'
        return self.size.strip() + '.toml'

    @property
    def folder(self) -> str:
        """Create a name for the training subfolder if none is given."""
        if self.name is None:
            options = ''.join([
                str(self.data),
                str(self.model),
                str(self.train)
            ])
            name = shake_128(options.encode()).hexdigest(4) + f'.{self.suffix}'
        else:
            name = self.name
        return str((Path(self.workdir) / name).resolve())

    @property
    def corpus(self) -> str:
        """Name of the corpus directory."""
        return str((Path(self.workdir) / self.files.corpus).resolve())

    @property
    def suffix(self) -> str:
        """Has of the tokenize setting to use in other names."""
        return shake_128(str(self.tokens).encode()).hexdigest(4)

    @property
    def encodings(self) -> str:
        """Create a name for the encodings directory."""
        if self.files.tokenizer is None:
            path = f'encodings.{self.suffix}'
        else:
            path = 'encodings.' + Path(self.files.tokenizer).stem
        return str((Path(self.workdir) / path).resolve())

    @property
    def tokenizer_file(self) -> str:
        """Create a tokenizer file name if none is given."""
        if self.files.tokenizer is None:
            file = f'tokenizer.{self.suffix}.json'
        else:
            file = Path(self.files.tokenizer).stem + '.json'
        return str((Path(self.workdir) / file).resolve())

    @property
    def checkpoint_file(self) -> str:
        """Name of the model checkpoint file."""
        return str((Path(self.folder) / self.files.checkpoint).resolve())

    @property
    def model_file(self) -> str:
        """Name of the file the model is saved to."""
        return str((Path(self.folder) / self.files.model).resolve())

    @property
    def weights_file(self) -> str:
        """Name of the file the model weights are saved to."""
        return str((Path(self.folder) / self.files.weights).resolve())

    @property
    def summary_file(self) -> str:
        """Name of the current training summary file."""
        return self.add_time(self.files.summary, 'toml')

    @property
    def log_file(self) -> str:
        """Name of the current training log file."""
        return self.add_time(self.files.log, 'log')

    @property
    def monitor_file(self):
        """Name of the current convergence tracking file."""
        return self.add_time(self.files.monitor, 'csv')

    @property
    def clean_files(self) -> str:
        """Naming schema for the cleand data files in the corpus directory."""
        return self.corpus + '/{}-{}.' + self.files.suffix.strip(' .')

    @property
    def encoded_files(self) -> str:
        """Naming schema for encoded files."""
        return self.encodings + '/{}'

    def add_time(self, subdir: str, suffix: str) -> str:
        """Generate new datetime file names if we're not resuming."""
        folder = Path(self.folder)
        files = sorted((folder / subdir).glob(f'*.{suffix}'))
        if files and self.resume:
            return str(files[-1].resolve())
        new = f'{self.start}.{suffix}'
        return str((folder / subdir / new).resolve())

    # ToDo: Resolve context<=seq_len in case of learnable positional encodings
