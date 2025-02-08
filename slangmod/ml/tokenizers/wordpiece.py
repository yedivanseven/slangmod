from tokenizers.pre_tokenizers import Sequence, WhitespaceSplit, Digits
from tokenizers.models import WordPiece
from tokenizers.decoders import WordPiece as WordPieceDecoder
from tokenizers.trainers import WordPieceTrainer
from ...config import config
from .common import normalizer
from .special import special
from .algo import Algo

__all__ = ['wordpiece']

model = WordPiece(
    vocab={},
    unk_token=special.unk.content,
    max_input_chars_per_word=config.tokens.max_len
)
trainer = WordPieceTrainer(
    vocab_size=config.tokens.vocab,
    min_frequency=config.tokens.min_freq,
    special_tokens=special.tokens,
    show_progress=config.progress
)
pre_tokenizer = Sequence([
    WhitespaceSplit(),
    Digits()
])
decoder = WordPieceDecoder(cleanup=True)

# Provide a ready-to-use instance of the Wordpiece tokenizer
wordpiece = Algo(
    special=special,
    tokenizer=model,
    trainer=trainer,
    normalizer=normalizer,
    pre_tokenizer=pre_tokenizer,
    decoder=decoder
)
