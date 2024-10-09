import tokenizers
from ...config import config

# Special tokens
pad = tokenizers.AddedToken(
    '[PAD]',
    single_word=True,
    special=True
)
unk = tokenizers.AddedToken(
    '[UNK]',
    single_word=True,
    special=True
)
eos = tokenizers.AddedToken(
    '[EOS]',
    single_word=True,
    special=True,
    normalized=True
)
special_tokens = [pad, unk, eos]

# Pipeline steps
normalizer = tokenizers.normalizers.Sequence([
    tokenizers.normalizers.Strip(),
    tokenizers.normalizers.NFD(),
    tokenizers.normalizers.StripAccents(),
    tokenizers.normalizers.Replace('\n\n', ' [EOS] '),
])
pre_tokenizer = tokenizers.pre_tokenizers.Sequence([
    tokenizers.pre_tokenizers.WhitespaceSplit(),
    tokenizers.pre_tokenizers.Metaspace(),
    tokenizers.pre_tokenizers.Punctuation(),
    tokenizers.pre_tokenizers.Digits(),
])
model = tokenizers.models.BPE(
    unk_token='[UNK]',
    end_of_word_suffix='</w>',
    fuse_unk=True
)
decoder = tokenizers.decoders.Metaspace()

# Actual tokenizer pipeline
tokenizer = tokenizers.Tokenizer(model)
tokenizer.add_tokens(special_tokens)
tokenizer.add_special_tokens(special_tokens)
tokenizer.normalizer = normalizer
tokenizer.pre_tokenizer = pre_tokenizer
tokenizer.decoder = decoder

# Trainer matching the Tokenizer
trainer = tokenizers.trainers.BpeTrainer(
    vocab_size=config.vocab_size,
    special_tokens=special_tokens,
    end_of_word_suffix='</w>'
)
