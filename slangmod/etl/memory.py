from ctypes import CDLL
from typing import overload
from functools import cached_property
from swak.misc import ArgRepr


class MemoryTrimmer[T, *Ts](ArgRepr):
    """Free up memory no longer used by NumPy arrays, PyTorch tensors, etc.

    These data structures cannot be reached by the python garbage collector
    and will block memory even if they cannot be referenced anymore from
    your code. In this case, memory can often be released by explicitly
    calling clib's ``malloc_trim``.

    Parameters
    ----------
    cdll: str, optional
        The name of the standard C dynamic-link library.
        Defaults to `libc.so.6`.

    """

    def __init__(self, cdll: str = 'libc.so.6') -> None:
        self.cdll = str(cdll).strip()
        super().__init__(self.cdll)

    @cached_property
    def libc(self) -> CDLL:
        """The loaded C library."""
        return CDLL(self.cdll)

    @overload
    def __call__(self) -> tuple[()]:
        ...

    @overload
    def __call__(self, arg: T) -> T:
        ...

    @overload
    def __call__(self, *args: *Ts) -> tuple[*Ts]:
        ...

    def __call__(self, *args):
        """Free up memory blocked by arrays the garbage collector can't reach.

        Parameters
        ----------
        *args
            To integrate anywhere in your code flow, instances can be called
            with any number of arguments, including none at all.

        Returns
        -------
        object or tuple
            An empty tuple if called with no arguments. When called with
            a single argument, that argument. When called with multiple
            arguments, a tuple of all of these arguments.

        """
        self.libc.malloc_trim(0)
        return args[0] if len(args) == 1 else args


# Provide a ready-to use instance of the MemoryTrimmer
trim_memory = MemoryTrimmer()
