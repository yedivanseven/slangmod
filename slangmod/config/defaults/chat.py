from swak.jsonobject import JsonObject
from swak.jsonobject.fields import Lower


class Chat(JsonObject):
    wrapper: Lower() = 'paragraph'
    generator: Lower() = 'top_p'
    k = 0.1
    p = 0.6
    system: str = 'Holmes was sitting in his armchair, facing the fireplace.'
    stop: str = 'Stop!'
