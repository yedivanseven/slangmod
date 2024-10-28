from pathlib import Path
from swak.jsonobject import JsonObject
from swak.jsonobject.fields import resolve


class Files(JsonObject):
    corpus: resolve = '/home/georg/Projects/slangmod/data/corpus'
    workdir: resolve = '/home/georg/Projects/slangmod/data'

    @property
    def tokenizer(self) -> str:
        return str((Path(self.workdir) / 'tokenizer.json').resolve())

    @property
    def checkpoint(self) -> str:
        return str((Path(self.workdir) / 'checkpoint.pt').resolve())

    @property
    def model(self) -> str:
        return str((Path(self.workdir) / 'model.pt').resolve())

    @property
    def config(self) -> str:
        return str((Path(self.workdir) / 'config.json').resolve())
