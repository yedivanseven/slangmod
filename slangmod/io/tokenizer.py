from tokenizers import Tokenizer
from swak.misc import ArgRepr
from ..config import config


class TokenizerSaver(ArgRepr):

    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.path = path

    def __call__(self, tokenizer: Tokenizer) -> tuple[()]:
        tokenizer.save(self.path)
        return ()


class TokenizerLoader(ArgRepr):

    def __init__(self, path: str) -> None:
        super().__init__(path)
        self.path = path

    def __call__(self) -> Tokenizer:
        return Tokenizer.from_file(self.path)


save_tokenizer = TokenizerSaver(config.tokenizer_file)
load_tokenizer = TokenizerLoader(config.tokenizer_file)
