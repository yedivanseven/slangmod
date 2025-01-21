import math
from collections.abc import Callable
import torch as pt
import torch.nn as ptn
from swak.pt.types import Tensor, Dtype, Device
from swak.pt.train import TestDataBase, TrainDataBase
from swak.pt.misc import LazyCatDim0
from swak.funcflow import Curry
from ..config import config, LiteralDevice, Devices
from .types import Batches

__all__ = [
    'TestData',
    'TrainData',
    'wrap_test_data',
    'wrap_train_data'
]


class TestData(TestDataBase):
    """Wraps test and validation data to provide batches over a sample.

    Parameters
    ----------
    seqs: Tensor
        A PyTorch tensor with dimensions (`N`, `S` + 1) of dtype `int64` or,
        equivalently, `long`, where `N` is the number of (padded)
        test/validation sequences and `S` is the (padded) sequence length.
        The "+1" is needed to provide the target for next-token prediction.
        Typically, this tensor will reside in CPU memory.
    device: str or  device, optional
        Torch device to push individual batches of data to. Defaults to "cpu",
        but will typically be "cuda".

    See Also
    --------
    TestSequenceFolder

    """

    def __init__(
            self,
            seqs: Tensor,
            device: Device | Devices | LiteralDevice = 'cpu',
            dtype: Dtype = pt.float
    ) -> None:
        self.seqs = seqs
        self.device = pt.device(device)
        self.dtype = dtype
        # ToDo: Remove mask once the reference model is gone.
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
        """Total number of test or validation sequences."""
        return self.seqs.size(0)

    @property
    def seq_len(self) -> int:
        """Length of (padded) test or validation sequences."""
        return self.seqs.size(1) - 1

    def sample(self, batch_size: int, max_n: int | None = None) -> Batches:
        """Reproducible sample of test or validation data for model evaluation.

        Parameters
        ----------
        batch_size: int
            The desired batch size. If the number of sequences is not integer
            divisible by that number, one of the batches will be smaller.
        max_n: int, optional
            The maximum number of sequences to provide in the sample, limited
            by how many there are in total. If not given, all sequences will
            be provided. Defaults to ``None``.

        Returns
        -------
        Iterator
            The items produced by the iterator are 2-tuples, with the first
            element being a one-tuple with a single batch of data with
            dimensions (`batch_size`, `seq_len`) and the second being a tensor
            of the same dimensions representing the target token ids, i.e.,
            the input shifted by one position.

        """
        n = self.n if max_n is None else min(max_n, self.n)
        batches = range(math.ceil(n / batch_size))
        seqs = self.seqs[:n]
        # ToDo: Return only sequences once the reference model is gone.
        return iter(
            (
                # Source sequence, attention mask, and is_causal flag
                (
                    seqs[batch * batch_size:(batch + 1) * batch_size, :-1].to(
                        self.device,
                        non_blocking=True
                    ),
                    self.mask,
                    None,  # unknown mask is None for testing
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
    """Wraps training data to provide batches and samples.

    Parameters
    ----------
    seqs: Tensor or LazyCatDim0
        A PyTorch tensor or an instance of `LazyCatDim0 <https://yedivanseven.
        github.io/swak/pt/misc.html#swak.pt.misc.LazyCatDim0>`_ with dimensions
        (`N`, `S` + `jitter`) of dtype `int64` or, equivalently, `long`, where
        `N` is the number of (padded) train sequences and `S` is the (padded)
        sequence length. Evidently, `jitter` must be at least 1 to provide the
        target for next-token prediction. Typically, this tensor will reside
        in CPU memory.
    shuffle: bool, optional
        Whether to randomize training data from one epoch to the next.
        If ``True``, sequences will be shifted by a random offset of up to
        `jitter` and the ordering of batches will be randomized as well.
        Defaults to ``True``.
    jitter: int, optional
        Maximum position index to randomly shift the start of the training
        sequences to for the next epoch, provided that `shuffle` is ``True``.
        Defaults to 1, which means that sequences will always start from the
        beginning. If sequences were not extended to account for this jitter,
        by using the ``TrainSequenceFolder``, for example, then the length
        of the training sequences will also randomly change from one epoch
        to the next.
    device: str or  device, optional
        Torch device to push individual batches of data to. Defaults to "cpu",
        but will typically be "cuda".

    See Also
    --------
    TrainSequenceFolder

    """

    def __init__(
            self,
            seqs: Tensor | LazyCatDim0,
            shuffle: bool = True,
            jitter: int = 1,
            device: Device | Devices | LiteralDevice = 'cpu',
            dtype: Dtype = pt.float
    ) -> None:
        self.seqs = seqs
        self.shuffle = shuffle
        self.jitter = self.__valid(jitter)
        self.device = pt.device(device)
        self.dtype = dtype
        # ToDo: Remove mask once the reference model is gone.
        self.mask = ptn.Transformer.generate_square_subsequent_mask(
            self.seq_len,
            device=self.device,
            dtype=dtype
        )

    def __valid(self, jitter: int) -> int:
        """Make sure that the given value is sane."""
        max_value = (self.seqs.size(1) - 2)
        if jitter > max_value:
            tmp = 'jitter ({}) may not be larger than {}!'
            msg = tmp.format(jitter, max_value)
            raise ValueError(msg)
        return jitter

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        return f'{cls}(n={self.n})'

    @property
    def n(self) -> int:
        """Total number of training sequences."""
        return self.seqs.size(0)

    @property
    def seq_len(self) -> int:
        """Length of (padded) training sequences."""
        return self.seqs.size(1) - self.jitter

    @property
    def _jumble(self) -> Callable[..., Tensor]:
        """Generate random or sorted batch numbers, depending on `shuffle`."""
        return pt.randperm if self.shuffle else pt.arange

    @property
    def _start(self) -> int:
        """Random or fixed start index of sequences, depending on `shuffle`."""
        return pt.randint(
            0,
            self.jitter,
            [1],
            device=self.seqs.device
        )[0] if self.shuffle else 0

    def sample(self, batch_size: int, max_n: int | None = None) -> Batches:
        """Reproducible sample of training data for model evaluation.

        Parameters
        ----------
        batch_size: int
            The desired batch size. If the number of sequences is not integer
            divisible by that number, one of the batches may be smaller.
        max_n: int, optional
            Approximate maximum number of sequences to provide in the sample.
            If enough are available, also the last batch will be filled up
            to the specified `batch_size`. If not given, all sequences will
            be provided. Defaults to ``None``.

        Returns
        -------
        Iterator
            The items produced by the iterator are 2-tuples, with the first
            element being a one-tuple with a single batch of data with
            dimensions (`batch_size`, `seq_len`) and the second being a tensor
            of the same dimensions representing the target token ids, i.e.,
            the input shifted by one position.

        Note
        ----
        To reproducibly compute a training loss or error, sequences will never
        be randomized in any way, regardless of `shuffle` and `jitter`.

        """
        n = self.n if max_n is None else min(max_n, self.n)
        batches = range(math.ceil(n / batch_size))
        # ToDo: Return only sequences once the reference model is gone.
        return iter(
            (
                # Source sequence, attention mask, and is_causal flag
                (
                    self.seqs[
                        batch * batch_size:(batch + 1) * batch_size,
                        :self.seq_len
                    ].to(
                        self.device,
                        non_blocking=True
                    ),
                    self.mask,
                    None,  # The unknown mask is None for training
                    True
                ),
                # Target sequence, shifted by one relative to the source
                self.seqs[
                    batch * batch_size:(batch + 1) * batch_size,
                    1:self.seq_len + 1
                ].to(
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
            _: int = 0
    ) -> tuple[int, Batches]:
        """Iterator over batches if training data for actual model training.

        Parameters
        ----------
        batch_size: int
            The desired batch size. If the number of sequences is not integer
            divisible by that number *and* `step_freq` is 1, one of the batches
            will be smaller than all others.
        step_freq: int, optional
            In case this number is > 1, all batches will have the exact
            `batch_size` such that losses accumulated over multiple batches
            can be appropriately scaled before taking an optimizer step.
            Defaults to 1.

        Returns
        -------
        n_batches: int
            Total number of batches the returned iterator will provide.
        batches: Iterator
            The items produced by the iterator are 2-tuples, with the first
            element being a one-tuple with a single batch of data with
            dimensions (`batch_size`, `seq_len`) and the second being a tensor
            of the same dimensions representing the target token ids, i.e.,
            the input shifted by one position.

        """
        start = self._start
        n_batches = self.adjust_batches_for(batch_size, step_freq)
        batches = self._jumble(n_batches, device=self.seqs.device)
        # ToDo: Return only sequences once the reference model is gone.
        return n_batches, iter(
            (
                # Source sequence, attention mask, and is_causal flag
                (
                    self.seqs[
                        batch * batch_size:(batch + 1) * batch_size,
                        start:start + self.seq_len
                    ].to(
                        self.device,
                        non_blocking=True
                    ),
                    self.mask,
                    None,  # The unknown mask is None for training
                    True
                ),
                # Target sequence, shifted by one relative to the source
                self.seqs[
                    batch * batch_size:(batch + 1) * batch_size,
                    start + 1:start + 1 + self.seq_len
                ].to(
                    self.device,
                    non_blocking=True
                )
            )
            for batch in batches
        )

# Provide partials of test and train data for convenience
wrap_train_data = Curry[TrainData](
    TrainData,
    shuffle=config.data.shuffle,
    jitter=config.data.jitter,
    device=config.data.device,
    dtype=config.data.dtype
)
wrap_test_data = Curry[TestData](
    TestData,
    device=config.data.device,
    dtype=config.data.dtype
)
