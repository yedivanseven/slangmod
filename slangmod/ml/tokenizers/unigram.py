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
from .common import PAD, UNK, EOS, normalizer
from .algo import Algo

__all__ = ['unigram']

model = Unigram([(PAD.content, 0.0), (UNK.content, 0.0)], unk_id=1)
trainer = UnigramTrainer(
    vocab_size=config.tokens.vocab,
    special_tokens=[PAD, UNK, EOS],
    shrinking_factor=config.tokens.shrink_factor,
    max_piece_length=config.tokens.max_length,
    n_sub_iterations=config.tokens.n_iter
)
pre_tokenizer = Sequence([
    WhitespaceSplit(),
    Digits(),
    MetaspacePreTokenizer()
])
decoder=MetaspaceDecoder()

unigram = Algo(
    model=model,
    trainer=trainer,
    normalizer=normalizer,
    pre_tokenizer=pre_tokenizer,
    decoder=decoder
)