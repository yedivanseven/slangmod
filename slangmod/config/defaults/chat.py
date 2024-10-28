from swak.jsonobject import JsonObject
from swak.jsonobject.fields import Lower
from ..enums import Styles


class Chat(JsonObject):
    style: Lower() = Styles.DIALOGUE
    generator: Lower() = 'top_p'
    k = 0.1
    p = 0.6
    system: str = 'Holmes was sitting in his armchair, facing the fireplace.'
    stop: str = 'Stop!'
