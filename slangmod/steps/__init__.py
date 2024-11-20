from .dry_run import dry_run
from .tokenize import tokenize
from .train import train
from .chat import chat
from .clean import clean_wiki40b

clean = clean_wiki40b

__all__ = [
    'dry_run',
    'clean',
    'clean_wiki40b',
    'tokenize',
    'train',
    'chat'
]

# ToDo: Make a "resume" step somehow!
