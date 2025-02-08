from collections.abc import Iterator
from swak.pt.types import Tensor

__all__ = [
    'Batch',
    'Batches',
    'Evaluation'
]

type Batch = tuple[tuple[Tensor, Tensor | None, Tensor | None, bool], Tensor]
type Batches = Iterator[Batch]
type Evaluation = tuple[float, float, float, float, float]
