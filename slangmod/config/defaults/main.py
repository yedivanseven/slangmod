import importlib.metadata as meta
from swak.jsonobject import JsonObject

PACKAGE = __name__.split('.')[0]
VERSION = meta.version(PACKAGE)


class Main(JsonObject):
    package: str = PACKAGE
    version: str = VERSION
    log_level: int = 10  # 10=debug, 20=info, 30=warning, 40=error, 50=critical
