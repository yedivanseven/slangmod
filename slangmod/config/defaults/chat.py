from swak.jsonobject import JsonObject
from swak.jsonobject.fields import Lower
from ..enums import Styles, Generators


class Chat(JsonObject):
    style: Lower() = Styles.SIMPLE
    generator: Lower() = Generators.TOP_P
    k = 0.1
    p = 0.6
    system: str = 'Holmes was sitting in his armchair, facing the fireplace.'
    stop: str = 'Stop!'
