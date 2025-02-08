from tokenizers.pre_tokenizers import Sequence, WhitespaceSplit, Digits
from tokenizers.models import BPE
from tokenizers.decoders import BPEDecoder
from tokenizers.trainers import BpeTrainer
from ...config import config
from .common import normalizer
from .special import special
from .algo import Algo

__all__ = ['bpe']

model = BPE(
    dropout=config.tokens.dropout,
    unk_token=special.unk.content,
    end_of_word_suffix=config.tokens.end_of_word_suffix,
    byte_fallback=False
)
trainer = BpeTrainer(
    vocab_size=config.tokens.vocab,
    min_frequency=config.tokens.min_freq,
    special_tokens=special.tokens,
    end_of_word_suffix=config.tokens.end_of_word_suffix,
    max_token_length=config.tokens.max_len,
    show_progress=config.progress
)
pre_tokenizer = Sequence([
    WhitespaceSplit(),
    Digits()
])
decoder = BPEDecoder(suffix=config.tokens.end_of_word_suffix)

# Provide a ready-to-use instance of the BPE tokenizer
bpe = Algo(
    special=special,
    tokenizer=model,
    trainer=trainer,
    normalizer=normalizer,
    pre_tokenizer=pre_tokenizer,
    decoder=decoder
)
