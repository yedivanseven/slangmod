from tokenizers import Tokenizer
from .algos import algo
from .common import special_tokens, normalizer, pre_tokenizer

__all__ = ['tokenizer']


# Actual tokenizer pipeline
tokenizer = Tokenizer(algo.model)
tokenizer.add_tokens(special_tokens)
tokenizer.add_special_tokens(special_tokens)
tokenizer.normalizer = normalizer
tokenizer.pre_tokenizer = pre_tokenizer
tokenizer.decoder = algo.decoder
