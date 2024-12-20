from .dry_run import dry_run
from .tokenize import tokenize
from .train import train
from .chat import chat
from .clean import clean


__all__ = [
    'dry_run',
    'clean',
    'tokenize',
    'train',
    'chat'
]

# ToDo: Make separate "encode" step to save lists of token ids
# ToDo: rewrite train step to read token ids from file
# ToDo: Make a "resume" step somehow!
from .encode import encode
