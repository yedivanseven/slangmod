import math
import random
import torch as pt
import torch.nn as ptn
from swak.pt.types import Tensor, Dtype, Device
from swak.pt.train import TestDataBase, TrainDataBase
from swak.funcflow import Curry
from ..config import config, LiteralDevice
from .types import Batches


class TestData(TestDataBase):

    def __init__(
            self,
            seqs: Tensor,
            device: Device | LiteralDevice,
            dtype: Dtype
    ) -> None:
        self.seqs = seqs
        self.device = pt.device(device)
        self.dtype = dtype
        self.mask = ptn.Transformer.generate_square_subsequent_mask(
            self.seq_len,
            device=self.device,
            dtype=dtype
        )
        self.__rand = pt.randperm(self.n, device=seqs.device, dtype=pt.long)

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        return f'{cls}(n={self.n})'

    @property
    def n(self) -> int:
        return self.seqs.shape[0]

    @property
    def seq_len(self) -> int:
        return self.seqs.shape[1] - 1

    def sample(self, batch_size: int, max_n: int | None = None) -> Batches:
        n = self.n if max_n is None else min(max_n, self.n)
        n_batches = math.ceil(n / batch_size)
        seqs = self.seqs[self.__rand[:n]]
        return iter(
            (
                (
                    seqs[batch * batch_size:(batch + 1) * batch_size, :-1].to(
                        self.device,
                        non_blocking=True
                    ),
                    self.mask,
                    None,
                    True
                ),
                seqs[batch * batch_size:(batch + 1) * batch_size, 1:].to(
                    self.device,
                    non_blocking=True
                )
            )
            for batch in range(n_batches)
        )


class TrainData(TrainDataBase):

    def __init__(
            self,
            seqs: Tensor,
            warmup: int,
            device: Device | LiteralDevice,
            dtype: Dtype
    ) -> None:
        self.seqs = seqs
        self.warmup = warmup
        self.device = pt.device(device)
        self.dtype = dtype
        self.mask = ptn.Transformer.generate_square_subsequent_mask(
            self.seq_len,
            device=self.device,
            dtype=dtype
        )
        self.__rand = pt.randperm(self.n, device=seqs.device, dtype=pt.long)

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        return f'{cls}(n={self.n})'

    @property
    def n(self) -> int:
        return self.seqs.shape[0]

    @property
    def seq_len(self) -> int:
        return self.seqs.shape[1] - 1

    def batch_lengths(
            self,
            batch_size: int,
            step_freq: int,
            epoch: int
    ) -> list[int]:
        n_batches = self.n_batches_of(batch_size, step_freq)
        batches = range(1, n_batches + 1)
        lengths = [max(1, int(self.seq_len * b / n_batches)) for b in batches]
        return [self.seq_len] + lengths[1:-1] if epoch == 1 else lengths

    def slices(
            self,
            batch_size: int,
            step_freq: int,
            epoch: int
    ) -> enumerate[tuple[slice, slice, int]]:
        bls = self.batch_lengths(batch_size, step_freq, epoch)
        starts = (random.randint(0, self.seq_len - bl) for bl in bls)
        slices = (
            (
                slice(start, start + bls[i]),
                slice(start + 1, start + 1 + bls[i]),
                bls[i]
            )
            for i, start in enumerate(starts)
        )
        return enumerate(slices)

    def sample(self, batch_size: int, max_n: int | None = None) -> Batches:
        n = self.n if max_n is None else min(max_n, self.n)
        n_batches = math.ceil(n / batch_size)
        seqs = self.seqs[self.__rand[:n]]
        return iter(
            (
                (
                    seqs[batch * batch_size:(batch + 1) * batch_size, :-1].to(
                        self.device,
                        non_blocking=True
                    ),
                    self.mask,
                    None,
                    True
                ),
                seqs[batch * batch_size:(batch + 1) * batch_size, 1:].to(
                    self.device,
                    non_blocking=True
                )
            )
            for batch in range(n_batches)
        )

    def __call__(
            self,
            batch_size: int,
            step_freq: int = 1,
            epoch: int = 0
    ) -> Batches:
        n = self.n_for(batch_size, step_freq)
        rand = pt.randperm(self.n, device=self.seqs.device, dtype=pt.long)
        seqs = self.seqs[rand[:n]]
        return iter(
            (
                # Source sequence, attention mask, and is_causal flag
                (
                    seqs[b * batch_size:(b + 1) * batch_size, :-1].to(
                        self.device,
                        non_blocking=True
                    ),
                    self.mask,  #[:bl, :bl],
                    None,  # The padding/unknown mask is None for training
                    True
                ),
                # Target sequence, shifted by one relative to the source
                seqs[b * batch_size:(b + 1) * batch_size, 1:].to(
                    self.device,
                    non_blocking=True
                )
            )
            for b, (src, tgt, bl) in self.slices(batch_size, step_freq, epoch)
        )


make_train_data = Curry[TrainData](
    TrainData,
    config.train.warmup,
    config.data.device,
    config.data.dtype
)
make_test_data = Curry[TestData](
    TestData,
    config.data.device,
    config.data.dtype
)
make_validation_data = Curry[TestData](
    TestData,
    config.data.device,
    config.data.dtype
)
