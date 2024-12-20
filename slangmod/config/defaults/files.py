from swak.jsonobject import JsonObject
from swak.jsonobject.fields import Lower, resolve


class Files(JsonObject):
    tokenizer: Lower() = 'tokenizer.json'
    checkpoint: Lower() = 'checkpoint.pt'
    model: Lower() = 'model.pt'
    config: Lower() = 'config.toml'
    suffix: str = 'parquet'
    sep: str = '-'
    train: str = 'train'
    test: str = 'test'
    validation: str = 'validation'
    wiki40b: resolve = '/home/georg/Projects/slangmod/data/wiki40b'
    gutenberg: resolve = '/home/georg/Projects/slangmod/data/gutenberg'
