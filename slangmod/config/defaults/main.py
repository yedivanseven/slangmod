import importlib.metadata as meta
import torch as pt
from swak.jsonobject import JsonObject
from swak.pt import device
from swak.pt.types import Dtype

PACKAGE = __name__.split('.')[0]
VERSION = meta.version(PACKAGE)


class Main(JsonObject):
    package: str = PACKAGE
    version: str = VERSION
    log_level: int = 10  # 10=debug, 20=info, 30=warning, 40=error, 50=critical
    books: str = '/home/georg/Projects/slangmod/data/books'
    workdir: str = '/home/georg/Projects/slangmod/data'
    vocab_size: int = 512
    context: int = 128
    frac_test: float = 0.1
    frac_validate: float = 0.1
    mod_dim: int = 8
    n_heads: int = 4
    n_layers: int = 1
    scale_grad_by_freq: bool = True
    dropout: float = 0.1
    bias: bool = True
    batch_size: int = 512
    label_smoothing: float = 0.0
    learning_rate: float = 0.0001
    gamma: float = 0.99
    max_epochs: int = 100
    warmup: int = 1
    patience: int = 2
    max_n: int = 1024 * 8

    @property
    def tokenizer_file(self) -> str:
        return '/' + self.workdir.strip(' /') + '/' + 'tokenizer.json'

    @property
    def checkpoint_file(self) -> str:
        return '/' + self.workdir.strip(' /') + '/' + 'checkpoint.pt'

    @property
    def model_file(self) -> str:
        return '/' + self.workdir.strip(' /') + '/' + 'model.pt'

    @property
    def dtype(self) -> Dtype:
        return pt.float32 if device.type == 'cpu' else pt.float16
