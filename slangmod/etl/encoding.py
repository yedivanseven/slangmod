import re
from swak.misc import ArgRepr
from ..config import config

__all__ = [
    'EncodingEnforcer',
    'enforce_encoding'
]


class EncodingEnforcer(ArgRepr):
    """Force a text into an encoding, replacing unrepresentable characters.

    Parameters
    ----------
    encoding: str
        The target encoding. Chose one from `the list of built-in codecs
        <https://docs.python.org/3/library/codecs.html#standard-encodings>`_.
    repl: str, optional
        The string to replace unrepresentable characters with.
        Defaults to a single space (" ").

    """

    pattern = re.compile(r'\\N{.+?}')

    def __init__(self, encoding: str, repl: str = ' ') -> None:
        self.encoding = encoding
        self.repl = repl
        super().__init__(encoding, repl)

    def __call__(self, text: str) -> str:
        """Replace unrepresentable characters in the given text.

        Parameters
        ----------
        text: str
            The text to force into the specified encoding by replacing
            unrepresentable characters.

        Returns
        -------
        str
            The text in the target encoding with unrepresentable characters
            replaced.

        """
        replaced = text.encode(
            self.encoding,
            'namereplace'
        ).decode(self.encoding)
        return self.pattern.sub(self.repl, replaced)


# Provide a ready-to use instance of the EncodingEnforcer
enforce_encoding = EncodingEnforcer(encoding=config.files.encoding)
