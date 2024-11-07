from swak.jsonobject import JsonObject
from swak.jsonobject.fields import Lower


class Files(JsonObject):
    tokenizer: Lower() = 'tokenizer.json'
    checkpoint: Lower() = 'checkpoint.pt'
    model: Lower() = 'model.pt'
    config: Lower() = 'config.toml'
