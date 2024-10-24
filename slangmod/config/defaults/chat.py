from swak.jsonobject import JsonObject
from swak.jsonobject.fields import Lower


class Chat(JsonObject):
    wrapper: Lower() = 'simple'
    generator: Lower() = 'top_p'
    k = 0.1
    p = 0.5
