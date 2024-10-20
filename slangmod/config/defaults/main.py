import importlib.metadata as meta
import torch as pt
from pathlib import Path
from swak.jsonobject import JsonObject
from swak.jsonobject.fields import Maybe, Lower, resolve
from swak.pt import device
from swak.pt.types import Dtype

PACKAGE = __name__.split('.')[0]
VERSION = meta.version(PACKAGE)


class Main(JsonObject):
    package: str = PACKAGE
    version: str = VERSION
    log_level: int = 10  # 10=debug, 20=info, 30=warning, 40=error, 50=critical
    books: resolve = '/home/georg/Projects/slangmod/data/books'
    workdir: resolve = '/home/georg/Projects/slangmod/data'
    tokenizer: Lower() = 'bpe'  # 'bpe' or 'wordpiece'
    vocab_size: int = 1024
    context: int = 256
    frac_test: float = 0.1
    frac_validate: float = 0.1
    mod_dim: int = 16
    n_heads: int = 2
    n_layers: int = 2
    scale_grad_by_freq: bool = True
    dropout: float = 0.1
    bias: bool = True
    batch_size: int = 256
    label_smoothing: float = 0.1
    learning_rate: float = 0.001
    gamma: float = 0.99
    max_epochs: int = 256
    warmup: int = 5
    patience: Maybe[int](int) = 5
    max_n: Maybe[int](int) = 4096

    @property
    def tokenizer_file(self) -> str:
        return str((Path(self.workdir) / 'tokenizer.json').resolve())

    @property
    def checkpoint_file(self) -> str:
        return str((Path(self.workdir) / 'checkpoint.pt').resolve())

    @property
    def model_file(self) -> str:
        return str((Path(self.workdir) / 'model.pt').resolve())

    @property
    def dtype(self) -> Dtype:
        return pt.float32 if device.type == 'cpu' else pt.bfloat16
