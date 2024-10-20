from abc import ABC, abstractmethod
from swak.pt.types import Device, Dtype, Tensor


class Positional(ABC):

    @abstractmethod
    def __init__(
            self,
            mod_dim: int,
            context: int,
            dtype: Dtype,
            device: Device,
    ) -> None:
        ...

    @property
    @abstractmethod
    def encodings(self) -> Tensor:
        ...
