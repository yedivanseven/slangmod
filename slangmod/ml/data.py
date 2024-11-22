import math
from collections.abc import Callable
import torch as pt
import torch.nn as ptn
from swak.pt.types import Tensor, Dtype, Device
from swak.pt.train import TestDataBase, TrainDataBase
from swak.funcflow import Curry
from ..config import config, LiteralDevice, Devices
from .types import Batches


class TestData(TestDataBase):

    def __init__(
            self,
            seqs: Tensor,
            shuffle: bool,
            device: Device | LiteralDevice,
            dtype: Dtype
    ) -> None:
        self.shuffle = shuffle
        self.device = pt.device(device)
        self.dtype = dtype
        self.seqs = self.pin(seqs.contiguous())
        self.__jumbled = self.jumble(self.n, device=seqs.device)
        self.mask = ptn.Transformer.generate_square_subsequent_mask(
            self.seq_len,
            device=self.device,
            dtype=dtype
        )

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        return f'{cls}(n={self.n})'

    @property
    def n(self) -> int:
        return self.seqs.size(0)

    @property
    def seq_len(self) -> int:
        return self.seqs.size(1) - 1

    @property
    def jumble(self) -> Callable[..., Tensor]:
        return pt.randperm if self.shuffle else pt.arange

    def pin(self, seqs: Tensor) -> Tensor:
        return seqs if self.device.type == Devices.CPU else seqs.pin_memory()

    def sample(self, batch_size: int, max_n: int | None = None) -> Batches:
        n = self.n if max_n is None else min(max_n, self.n)
        batches = range(math.ceil(n / batch_size))
        seqs = self.seqs[self.__jumbled[:n]]
        return iter(
            (
                # Source sequence, attention mask, and is_causal flag
                (
                    seqs[batch * batch_size:(batch + 1) * batch_size, :-1].to(
                        self.device,
                        non_blocking=True
                    ),
                    self.mask,
                    None,  # The padding/unknown mask is None for testing
                    True
                ),
                # Target sequence, shifted by one relative to the source
                seqs[batch * batch_size:(batch + 1) * batch_size, 1:].to(
                    self.device,
                    non_blocking=True
                )
            )
            for batch in batches
        )


class TrainData(TrainDataBase):

    def __init__(
            self,
            seqs: Tensor,
            stride: int,
            shuffle: bool,
            jitter: int,
            device: Device | LiteralDevice,
            dtype: Dtype
    ) -> None:
        self.stride = stride
        self.shuffle = shuffle
        self.jitter = min(max(1, jitter), stride)
        self.device = pt.device(device)
        self.dtype = dtype
        self.seqs = self.pin(seqs.contiguous())
        self.__jumbled = self.jumble(self.n, device=seqs.device)
        self.__start = self.start
        self.mask = ptn.Transformer.generate_square_subsequent_mask(
            self.seq_len,
            device=self.device,
            dtype=dtype
        )

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        return f'{cls}(n={self.n})'

    @property
    def n(self) -> int:
        return self.seqs.size(0)

    @property
    def seq_len(self) -> int:
        return self.seqs.size(1) - self.stride

    @property
    def jumble(self) -> Callable[..., Tensor]:
        return pt.randperm if self.shuffle else pt.arange

    @property
    def start(self) -> int:
        return pt.randint(
            0,
            self.jitter,
            [1],
            device=self.seqs.device
        ) if self.shuffle else 0

    def pin(self, seqs: Tensor) -> Tensor:
        return seqs if self.device.type == Devices.CPU else seqs.pin_memory()

    def sample(self, batch_size: int, max_n: int | None = None) -> Batches:
        n = self.n if max_n is None else min(max_n, self.n)
        batches = range(math.ceil(n / batch_size))
        seqs = self.seqs[
           self.__jumbled[:n],
           self.__start:self.__start + self.seq_len + 1
        ]
        return iter(
            (
                # Source sequence, attention mask, and is_causal flag
                (
                    seqs[batch * batch_size:(batch + 1) * batch_size, :-1].to(
                        self.device,
                        non_blocking=True
                    ),
                    self.mask,
                    None,  # The padding/unknown mask is None for training
                    True
                ),
                # Target sequence, shifted by one relative to the source
                seqs[batch * batch_size:(batch + 1) * batch_size, 1:].to(
                    self.device,
                    non_blocking=True
                )
            )
            for batch in batches
        )

    def __call__(
            self,
            batch_size: int,
            step_freq: int = 1,
            epoch: int = 0
    ) -> tuple[int, Batches]:
        jumbled = self.jumble(self.n, device=self.seqs.device)
        start = self.start
        seqs = self.seqs[jumbled, start:start + self.seq_len + 1]
        n_batches = self.adjust_batches_for(batch_size, step_freq)
        return n_batches, iter(
            (
                # Source sequence, attention mask, and is_causal flag
                (
                    seqs[batch * batch_size:(batch + 1) * batch_size, :-1].to(
                        self.device,
                        non_blocking=True
                    ),
                    self.mask,
                    None,  # The padding/unknown mask is None for training
                    True
                ),
                # Target sequence, shifted by one relative to the source
                seqs[batch * batch_size:(batch + 1) * batch_size, 1:].to(
                    self.device,
                    non_blocking=True
                )
            )
            for batch in range(n_batches)
        )


make_train_data = Curry[TrainData](
    TrainData,
    stride=config.data.stride,
    shuffle=config.data.shuffle,
    jitter=config.data.jitter,
    device=config.data.device,
    dtype=config.data.dtype
)
make_test_data = Curry[TestData](
    TestData,
    shuffle=config.data.shuffle,
    device=config.data.device,
    dtype=config.data.dtype
)
