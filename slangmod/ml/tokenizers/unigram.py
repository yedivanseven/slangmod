from tokenizers.pre_tokenizers import (
    Sequence,
    WhitespaceSplit,
    Digits,
    Metaspace as MetaspacePreTokenizer
)
from tokenizers.models import Unigram
from tokenizers.decoders import Metaspace as MetaspaceDecoder
from tokenizers.trainers import UnigramTrainer
from ...config import config
from .common import normalizer
from .special import special
from .algo import Algo

__all__ = ['unigram']

model = Unigram(special.unigram_vocab, special.unk_id, False)
trainer = UnigramTrainer(
    vocab_size=config.tokens.vocab,
    special_tokens=special.tokens,
    shrinking_factor=config.tokens.shrink_factor,
    unk_token=special.unk.content,
    max_piece_length=config.tokens.max_len,
    n_sub_iterations=config.tokens.n_iter,
    show_progress=config.progress
)
pre_tokenizer = Sequence([
    WhitespaceSplit(),
    Digits(),
    MetaspacePreTokenizer()
])
decoder = MetaspaceDecoder()

# Provide a ready-to-use instance of the Unigram tokenizer
unigram = Algo(
    special=special,
    tokenizer=model,
    trainer=trainer,
    normalizer=normalizer,
    pre_tokenizer=pre_tokenizer,
    decoder=decoder
)
