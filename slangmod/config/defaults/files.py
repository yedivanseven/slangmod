from swak.jsonobject import JsonObject
from swak.jsonobject.fields import Lower, resolve


class Files(JsonObject):
    summary = 'summary'
    monitor = 'convergence'
    log = 'debug.log'

    tokenizer: Lower() = 'tokenizer.json'
    checkpoint: Lower() = 'checkpoint.pt'
    model: Lower() = 'model.pt'
    suffix: str = 'parquet'
    train: str = 'train'
    test: str = 'test'
    column: str = 'text'
    validation: str = 'validation'
    wiki40b: resolve = '/home/georg/Projects/slangmod/data/wiki40b'
    gutenberg: resolve = '/home/georg/Projects/slangmod/data/gutenberg'

    @property
    def types(self) -> tuple[str, str, str]:
        return self.train, self.test, self.validation
