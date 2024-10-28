from tokenizers.pre_tokenizers import Sequence, WhitespaceSplit, Digits
from tokenizers.models import BPE
from tokenizers.decoders import BPEDecoder
from tokenizers.trainers import BpeTrainer
from ...config import config
from .common import PAD, UNK, EOS, normalizer
from .algo import Algo

__all__ = ['bpe']

model = BPE(
    dropout=config.tokens.dropout,
    unk_token=UNK.content,
    end_of_word_suffix='</w>',
    fuse_unk=True
)
trainer = BpeTrainer(
    vocab_size=config.tokens.vocab,
    min_frequency=config.tokens.min_frequency,
    special_tokens=[PAD, UNK, EOS],
    end_of_word_suffix='</w>',
    max_token_length=config.tokens.max_length
)
pre_tokenizer = Sequence([
    WhitespaceSplit(),
    Digits()
])
decoder=BPEDecoder(suffix='</w>')

bpe = Algo(
    model=model,
    trainer=trainer,
    normalizer=normalizer,
    pre_tokenizer=pre_tokenizer,
    decoder=decoder
)
