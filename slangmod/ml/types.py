from collections.abc import Iterator
from swak.pt.types import Tensor

type Batch = tuple[tuple[Tensor, Tensor | None, Tensor | None, bool], Tensor]
type Batches = Iterator[Batch]
type Validation = tuple[float, float, float, float]
