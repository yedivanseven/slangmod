import os
import sys


def log_corpus(corpus: list[str]) -> str:
    formatted = '\n'.join(corpus)
    if corpus:
        return f'Found files:\n{formatted}\nlogging from module'
    sys.exit(os.EX_OK)
