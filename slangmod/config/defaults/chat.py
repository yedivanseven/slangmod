from pathlib import Path
from swak.jsonobject import JsonObject
from swak.jsonobject.fields import Lower
from ..enums import Styles, Generators


def read_system_prompt(text: str) -> str:
    """Read the system prompt from file if present, simply forward it if not.

    Parameters
    ----------
    text: str
        Full path to text file with system prompt in it or the system
        or the system prompt itself.

    Returns
    -------
    str
        The contents of the `text` file if it was found or the given `text`
        if it was not.

    """
    path = Path(text.strip())
    try:
        text_could_be_a_file = path.exists() and path.is_file()
    except OSError:
        text_could_be_a_file = False
    if text_could_be_a_file:
        with path.open() as file:
            text = file.read()
        return text
    return text


class Chat(JsonObject):
    generator: Lower() = Generators.GREEDY
    max_tokens: int = 256
    temperature: float = 1.0
    k: float = 0.1
    p: float = 0.8
    width: int = 8
    boost: float = 1.0
    style: Lower() = Styles.SPACE
    user: str = 'USR'
    bot: str = 'BOT'
    stop: str = 'Stop!'
    system: read_system_prompt = ''
