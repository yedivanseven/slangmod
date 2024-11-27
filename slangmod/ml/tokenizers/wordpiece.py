from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Sequence, WhitespaceSplit, Digits
from tokenizers.models import WordPiece
from tokenizers.decoders import WordPiece as WordPieceDecoder
from tokenizers.trainers import WordPieceTrainer
from ...config import config
from .common import PAD, UNK, EOS, normalizer
from .algo import Algo

__all__ = ['wordpiece']

model = WordPiece(
    unk_token=UNK.content,
    max_input_chars_per_word=config.tokens.max_length
)
trainer = WordPieceTrainer(
    vocab_size=config.tokens.vocab,
    min_frequency=config.tokens.min_frequency,
    special_tokens=[PAD, UNK, EOS],
    continuing_subword_prefix='##'
)
pre_tokenizer = Sequence([
    WhitespaceSplit(),
    Digits()
])
decoder = WordPieceDecoder(prefix='##', cleanup=True)

wordpiece = Algo(
    tokenizer=Tokenizer(model),
    trainer=trainer,
    normalizer=normalizer,
    pre_tokenizer=pre_tokenizer,
    decoder=decoder
)
