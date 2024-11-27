from tokenizers import Tokenizer
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
from .common import PAD, UNK, EOS, SPECIAL_TOKENS, normalizer
from .algo import Algo

__all__ = ['unigram']

vocab = [(PAD.content, 0.0), (UNK.content, 0.0), (EOS.content, 0.0)]
model = Unigram(vocab, unk_id=1)
trainer = UnigramTrainer(
    vocab_size=config.tokens.vocab,
    special_tokens=SPECIAL_TOKENS,
    shrinking_factor=config.tokens.shrink_factor,
    unk_token=UNK.content,
    max_piece_length=config.tokens.max_length,
    n_sub_iterations=config.tokens.n_iter
)
pre_tokenizer = Sequence([
    WhitespaceSplit(),
    Digits(),
    MetaspacePreTokenizer()
])
decoder = MetaspaceDecoder()

unigram = Algo(
    tokenizer=Tokenizer(model),
    trainer=trainer,
    normalizer=normalizer,
    pre_tokenizer=pre_tokenizer,
    decoder=decoder
)
