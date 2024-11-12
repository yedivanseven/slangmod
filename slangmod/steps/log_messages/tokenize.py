def log_corpus(corpus: list[str]) -> str:
    formatted = '\n'.join(corpus)
    return f'Found files:\n{formatted}\nlogging from module'
