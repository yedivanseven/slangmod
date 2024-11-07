from swak.jsonobject import JsonObject
from swak.jsonobject.fields import Lower
from ..enums import Styles, Generators


class Chat(JsonObject):
    style: Lower() = Styles.SPACE
    generator: Lower() = Generators.TOP_P
    k: float = 0.1
    p: float = 0.6
    width: int = 4
    user: str = 'user'
    bot: str = 'bot'
    stop: str = 'Stop!'
    system: str = 'Holmes was sitting in his armchair, facing the fireplace.'
