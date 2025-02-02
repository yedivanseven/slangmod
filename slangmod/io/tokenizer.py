from pathlib import Path
from swak.misc import ArgRepr
from tokenizers import Tokenizer
from ..config import config
from ..ml import Algo, tokenizer

__all__ = [
    'TokenizerSaver',
    'TokenizerLoader',
    'save_tokenizer',
    'load_tokenizer'
]


class TokenizerSaver(ArgRepr):
    """Convenience wrapper around a Tokenizer's or Algo's ``save`` method.

    Parameters
    ----------
    path: str
        Path (including file name) to save the tokenizer to. May include any
        number of string placeholders (i.e., pairs of curly brackets) that will
        be interpolated when instances are called. Defaults to the current
        working directory of the python interpreter.
    create: bool, optional
        What to do if the directory where the tokenizer should be saved does
        not exist. Defaults to ``False``.

    """

    def __init__(self, path: str = '', create: bool = False) -> None:
        self.path = str(path).strip()
        self.create = create
        super().__init__(self.path, create)

    def __call__(self, algo: Tokenizer | Algo, *parts: str) -> tuple[()]:
        """Save a Tokenizer or Algo to file.

        Parameters
        ----------
        algo: Tokenizer or Algo
            The tokenizer to save.
        *parts: str, optional
            Fragments that will be interpolated into the `path` string given at
            instantiation. Obviously, there must be at least as many as there
            are placeholders in the `path`.

        Returns
        -------
        tuple
            An empty tuple.

        """
        path = Path(self.path.format(*parts).strip())
        file = str(path.resolve())
        if self.create:
            path.parent.mkdir(parents=True, exist_ok=True)
        algo.save(file)
        return ()


class TokenizerLoader[T: (Algo, Tokenizer)](ArgRepr):
    """Load a previously saved Tokenizer or Algo from file.

    Parameters
    ----------
    algo: Tokenizer or Algo
        A fresh, trained, or tainted instance of a tokenizer or an Algo.
    path: str, optional
        Full or partial path to the model to load. If not fully specified here,
        it can be completed on calling the instance. Defaults to the current
        working directory of the python interpreter.

    """

    def __init__(self, algo: T, path: str = '') -> None:
        self.algo = algo
        self.path = str(path).strip()
        super().__init__(self.path)

    def __call__(self, path: str = '') -> T:
        """Load a previously saved Tokenizer or Algo from file

        Parameters
        ----------
        path: str, optional
            Path (including file name) to the file to load. If it starts
            with a backslash, it will be interpreted as absolute, if not, as
            relative to the `path` specified at instantiation. Defaults to an
            empty string, which results in an unchanged `path`.

        Returns
        -------
        Tokenizer or Algo
            A new instance of the same type as the `algo` provided at
            instantiation with its internal parameters set to what was read
            from file.

        """
        path = Path(self.path) / str(path).strip()
        file = str(path.resolve())
        return self.algo.from_file(file)


# Provide read-to-use instances of both the Save and the Loader
save_tokenizer = TokenizerSaver(config.tokenizer_file, True)
load_tokenizer = TokenizerLoader(tokenizer, config.tokenizer_file)
