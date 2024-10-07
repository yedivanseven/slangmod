from tokenizers import Tokenizer
from tokenizers.trainers import Trainer


class TokenizerTrainer:

    def __init__(
            self,
            tokenizer: Tokenizer,
            trainer: Trainer
    ) -> None:
        self.tokenizer = tokenizer
        self.trainer = trainer

    def __repr__(self) -> str:
        cls = self.tokenizer.model.__class__.__name__
        return f'{cls}(...)'

    def __call__(self, files: list[str]) -> Tokenizer:
        self.tokenizer.train(files, self.trainer)
        return self.tokenizer
