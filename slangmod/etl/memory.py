import ctypes


class MemoryTrimmer[T]:

    def __init__(self) -> None:
        self.libc = ctypes.CDLL('libc.so.6')

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        return f"{cls}('libc.so.6')"

    def __call__(self, *args: T) -> T:
        self.libc.malloc_trim(0)
        return args[0] if len(args) == 1 else args


trim_memory = MemoryTrimmer()
