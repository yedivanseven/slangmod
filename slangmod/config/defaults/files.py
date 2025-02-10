from swak.jsonobject import JsonObject
from swak.jsonobject.fields import Lower, resolve, Maybe


class Files(JsonObject):
    train = 'train'
    test = 'test'
    validation = 'validation'
    summary = 'summary'
    monitor = 'convergence'
    corpus = 'corpus'
    log = 'logs'
    chat = 'chats'

    raw: resolve = '.'
    suffix: str = 'parquet'
    column: str = 'text'
    min_doc_len: int = 1
    cleaners: list[str] = []
    encoding: Lower() = 'cp1252'

    tokenizer: Maybe[str](Lower()) = None
    checkpoint: Lower() = 'checkpoint.pt'
    model: Lower() = 'model.pt'
    weights: Lower() = 'weights.pt'

    @property
    def types(self) -> tuple[str, str, str]:
        """Tuple of required file types."""
        return self.train, self.test, self.validation
