from tokenizers.models import Model, BPE, WordPiece
from tokenizers.decoders import (
    Decoder,
    BPEDecoder,
    WordPiece as WordPieceDecoder
)
from tokenizers.trainers import Trainer, BpeTrainer, WordPieceTrainer
from swak.misc import ArgRepr
from ...config import config, Tokenizers
from .common import special_tokens

__all__ = [
    'BPETokenizer',
    'WordPieceTokenizer',
    'bpe',
    'wordpiece',
    'algo',
]


class BPETokenizer(ArgRepr):

    def __init__(
            self,
            vocab_size: int,
            special_tokens: list[str],
            dropout: float = 0.0
    ) -> None:
        super().__init__(vocab_size, special_tokens, dropout)
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.dropout = dropout

    @property
    def model(self) -> Model:
        return BPE(
            dropout=self.dropout,
            unk_token='[UNK]',
            end_of_word_suffix='</w>',
            fuse_unk=True
        )

    @property
    def decoder(self) -> Decoder:
        return BPEDecoder(suffix='</w>')

    @property
    def trainer(self) -> Trainer:
        return BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
            end_of_word_suffix='</w>'
        )


class WordPieceTokenizer(ArgRepr):

    def __init__(self, vocab_size: int, special_tokens: list[str]) -> None:
        super().__init__(vocab_size, special_tokens)
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens

    @property
    def model(self) -> Model:
        return WordPiece(unk_token='[UNK]')

    @property
    def decoder(self) -> Decoder:
        return WordPieceDecoder()

    @property
    def trainer(self) -> Trainer:
        return WordPieceTrainer(
            vocab_size=self.vocab_size,
            special_tokens=self.special_tokens,
            continuing_subword_prefix='##'
        )


bpe = BPETokenizer(
    config.tokenizer.vocab_size,
    special_tokens,
    config.tokenizer.dropout
)
wordpiece = WordPieceTokenizer(
    config.tokenizer.vocab_size,
    special_tokens
)
algo = {
    Tokenizers.BPE: bpe,
    Tokenizers.WORDPIECE :wordpiece
}[config.tokenizer.algo]
