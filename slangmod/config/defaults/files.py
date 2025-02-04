from swak.jsonobject import JsonObject
from swak.jsonobject.fields import Lower, resolve
from ..enums import Cleaners


class Files(JsonObject):
    train = 'train'
    test = 'test'
    validation = 'validation'
    summary = 'summary'
    monitor = 'convergence'
    corpus = 'corpus'
    encodings = 'encodings'
    log = 'debug.log'

    raw = '/home/georg/Projects/slangmod/data/wiki40b'
    suffix: str = 'parquet'
    column: str = 'text'
    min_doc_len: int = 32
    cleaners: list[str] = [
        Cleaners.WIKI40B,
        Cleaners.QUOTES,
        Cleaners.ENCODING
    ]
    encoding: Lower() = 'cp1252'

    tokenizer: Lower() = 'tokenizer.json'
    checkpoint: Lower() = 'checkpoint.pt'
    model: Lower() = 'model.pt'
    wiki40b: resolve = '/home/georg/Projects/slangmod/data/wiki40b'
    gutenberg: resolve = '/home/georg/Projects/slangmod/data/gutenberg'

    @property
    def types(self) -> tuple[str, str, str]:
        """Tuple of required file types."""
        return self.train, self.test, self.validation
