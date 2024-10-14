from tokenizers import Tokenizer
from tokenizers.trainers import Trainer
from swak.misc import ArgRepr
from .tokenizer import tokenizer
from .algos import algo

__all__ = [
    'TokenizerTrainer',
    'train_tokenizer'
]


class TokenizerTrainer(ArgRepr):

    def __init__(
            self,
            tokenizer: Tokenizer,
            trainer: Trainer
    ) -> None:
        super().__init__(tokenizer, trainer)
        self.tokenizer = tokenizer
        self.trainer = trainer

    def __repr__(self) -> str:
        cls = self.tokenizer.model.__class__.__name__
        return f'{cls}(...)'

    def __call__(self, files: list[str]) -> Tokenizer:
        self.tokenizer.train(files, self.trainer)
        return self.tokenizer


train_tokenizer = TokenizerTrainer(tokenizer, algo.trainer)
