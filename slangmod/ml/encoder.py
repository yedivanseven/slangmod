from tokenizers import Tokenizer


class Encoder:

    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        return f'{cls}(tokenizer)'

    def __call__(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids
