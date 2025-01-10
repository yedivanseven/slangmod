from typing import Any
import torch.nn.functional as ptnf
from swak.misc import ArgRepr
from swak.pt.types import Tensor
from ..config import config
from .exceptions import ValidationErrors

__all__ = [
    'SequenceFolder',
    'TestSequenceFolder',
    'TrainSequenceFolder',
    'fold_train',
    'fold_test'
]


class SequenceFolder(ArgRepr):

    def __init__(self, seq_len: int, stride: int = 0, jitter: int = 0) -> None:
        self.seq_len = seq_len
        self.stride = max(0, min(stride, seq_len))
        self.jitter = min(self.stride, jitter)
        super().__init__(seq_len, self.stride, self.jitter)

    @property
    def width(self) -> int:
        if self.stride > 0:
            return self.seq_len + self.jitter
        return self.seq_len + 1

    def clamp(self, length: int) -> int:
        if self.stride > 0:
            return max(self.jitter + 1, length)
        return max(0, length - 2)

    def extra_rows_for(self, length: int) -> int:
        if self.stride > 0:
            return (self.clamp(length) - self.jitter - 1) // self.stride
        return self.clamp(length) // self.seq_len

    def padded_to(self, length: int) -> int:
        if self.stride > 0:
            return self.width + self.extra_rows_for(length) * self.stride
        return self.width + self.extra_rows_for(length) * self.seq_len

    def missing_for(self, length: int) -> int:
        return self.padded_to(length) - length

    def pad(self, sequence: Tensor) -> Tensor:
        n_pad = self.missing_for(sequence.size(0))
        return ptnf.pad(sequence, (0, n_pad), value=0)

    def __call__(self, sequence: Tensor) -> Tensor:
        if self.stride > 0:
            return self.pad(sequence).unfold(0, self.width, self.stride)
        return self.pad(sequence).unfold(0, self.width, self.seq_len)


class TestSequenceFolder(ArgRepr):
    """Warp and right-pad a single sequence into many with specified length.

    Parameters
    ----------
    seq_len: int
        The target sequence length. Because, in next-token prediction, the
        target sequence is the source offset by one, the output tensor size
        will be one more than this number in its second and last dimension.
    pad_id: int, optional
        Integer index to pad sequences with so that all parts have the same
        length,that is, `seq_len` + 1. Defaults to 0. Make sure that this
        index is consistently used also for training embeddings and in your
        loss function!

    Raises
    ------
    TypeError
        If `seq_len` is not an integer.
    ValueError
        It `seq_len` is smaller than 2.

    """

    def __init__(self, seq_len: int, pad_id: int = 0) -> None:
        super().__init__(seq_len, pad_id)
        self.seq_len = self.__valid(self.__typed(seq_len))
        self.pad_id = pad_id

    @property
    def width(self) -> int:
        """Size of the output tensor in its second and last dimension."""
        return self.seq_len + 1

    @staticmethod
    def __typed(seq_len: Any) -> int:
        """Check that the given sequence length is an integer."""
        if not isinstance(seq_len, int):
            cls = type(seq_len).__name__
            msg = f'seq_len must be integer, not {cls}!'
            raise TypeError(msg)
        return seq_len

    @staticmethod
    def __valid(seq_len: int) -> int:
        """Check that the given sequence length is sane."""
        if seq_len < 2:
            msg = f'seq_len must be at least 2, unlike {seq_len}!'
            raise ValueError(msg)
        return seq_len

    @staticmethod
    def _clamped(length: int) -> int:
        """Sequence length to compute the number of extra rows needed."""
        return max(0, length - 2)

    def _extra_rows_for(self, length: int) -> int:
        """The number of rows to add to accommodate the folded sequence."""
        return self._clamped(length) // self.seq_len

    def _padded_to(self, length: int) -> int:
        """Determine the length to pad to such that folding can work."""
        return self.width + self._extra_rows_for(length) * self.seq_len

    def _missing_for(self, length: int) -> int:
        """How many padding indices are needed to reach the required length?"""
        return self._padded_to(length) - length

    def _pad(self, sequence: Tensor) -> Tensor:
        """Pad input sequence to the required length with the padding index."""
        n_pad = self._missing_for(sequence.size(0))
        return ptnf.pad(sequence, (0, n_pad), value=self.pad_id)

    def __call__(self, sequence: Tensor) -> Tensor:
        """Warp and right-pad a single sequence into many of the given length.

        Each row of the output can then conveniently be split into source
        (``row[:-1]``) and target (``row[1:]``) for next-token prediction such
        that both have the given `seq_len`.

        Parameters
        ----------
        sequence: Tensor
            The input sequence to warp and pad. Must be a 1-dimensional tensor.

        Returns
        -------
        Tensor
            Output tensor of size `seq_len` + 1 in its second and last
            dimension and as many rows as needed to accommodate all elements
            in its first dimension.

        Notes
        -----
        Sequences that are folded into more than one row will never have any
        row that has fewer than 2 non-padding entries because these would be
        useless in evaluating next-token prediction. However, empty sequences
        or sequences of length 1 will still be padded to a tensor of sizes 1
        and `seq_len` + 1 in its first and second dimension, respectively.
        Consequently, they should be filtered out beforehand!

        """
        return self._pad(sequence).unfold(0, self.width, self.seq_len)


class TrainSequenceFolder(ArgRepr):
    """Warp and right-pad a single sequence into many with specified length.

    Parameters
    ----------
    seq_len: int
        The target sequence length. Because, in next-token prediction, the
        target sequence is the source offset by one, the output tensor size
        will be one more than this number in its second and last dimension.
    pad_id: int, optional
        Integer index to pad sequences with so that all parts have the same
        length,that is, `seq_len` + 1. Defaults to 0. Make sure that this
        index is consistently used also for training embeddings and in your
        loss function!

    """

    def __init__(
            self,
            seq_len: int,
            pad_id: int = 0,
            overlap: float = 0.0,
            jitter: int = 1
    ) -> None:
        self.pad_id = pad_id
        seq_len, overlap, jitter = self.__typed(seq_len, overlap, jitter)
        overlap = round(seq_len * overlap) if overlap < 1.0 else overlap
        self.seq_len, self.overlap, self.jitter = self.__valid(
            seq_len, overlap, jitter
        )
        super().__init__(self.seq_len, pad_id, self.overlap, self.jitter)

    @property
    def width(self) -> int:
        """Size of the output tensor in its second and last dimension."""
        return self.seq_len + self.jitter

    @property
    def stride(self) -> int:
        """The number of elements between the two consecutive sequences."""
        return self.seq_len - self.overlap

    @staticmethod
    def __typed(
            seq_len: Any,
            overlap: Any,
            jitter: Any
    ) -> tuple[int, int, int]:
        errors = []
        if not isinstance(seq_len, int):
            cls = type(seq_len).__name__
            msg = f'seq_len must be integer, not {cls}!'
            errors.append(TypeError(msg))
        if not isinstance(overlap, int | float):
            cls = type(overlap).__name__
            msg = f'overlap must be integer, not {cls}!'
            errors.append(TypeError(msg))
        if not isinstance(jitter, int):
            cls = type(jitter).__name__
            msg = f'jitter must be integer, not {cls}!'
            errors.append(TypeError(msg))

        if errors:
            raise ValidationErrors('Type validation failed', errors)

        return seq_len, overlap, jitter

    @staticmethod
    def __valid(
            seq_len: int,
            overlap: int,
            jitter: int
    ) -> tuple[int, int, int]:
        """Check that sequence length, stride, and jitter are all sane."""
        errors = []
        if seq_len < 2:
            msg = f'seq_len must be at least 2, unlike {seq_len}!'
            errors.append(ValueError(msg))
        if overlap < 0:
            msg = f'overlap must be positive, unlike {overlap}!'
            errors.append(ValueError(msg))
        if overlap >= seq_len:
            msg = 'overlap (={}) cannot be larger than seq_len - 1 (={})!'
            errors.append(ValueError(msg.format(overlap, seq_len - 1)))
        if jitter < 1:
            msg = f'jitter must be at least 1, unlike {jitter}!'
            errors.append(ValueError(msg))
        if jitter > seq_len - overlap:
            msg = 'jitter (={}) cannot be larger than seq_len - overlap (={})!'
            errors.append(ValueError(msg.format(jitter, seq_len - overlap)))

        if errors:
            raise ValidationErrors('Value validation failed', errors)

        return seq_len, overlap, jitter

    def _clamped(self, length: int) -> int:
        """Sequence length to compute the number of extra rows needed."""
        return max(self.jitter + 1, length)

    def _extra_rows_for(self, length: int) -> int:
        """The number of rows to add to accommodate the folded sequence."""
        return (self._clamped(length) - self.jitter - 1) // self.stride

    def _padded_to(self, length: int) -> int:
        """Determine the length to pad to such that folding can work."""
        return self.width + self._extra_rows_for(length) * self.stride

    def _missing_for(self, length: int) -> int:
        """How many padding indices are needed to reach the required length?"""
        return self._padded_to(length) - length

    def _pad(self, sequence: Tensor) -> Tensor:
        """Pad input sequence to the required length with the padding index."""
        n_pad = self._missing_for(sequence.size(0))
        return ptnf.pad(sequence, (0, n_pad), value=self.pad_id)

    def __call__(self, sequence: Tensor) -> Tensor:
        """Warp and right-pad a single sequence into many of the given length.

        Each row of the output can then conveniently be split into source
        (``row[:-1]``) and target (``row[1:]``) for next-token prediction such
        that both have the given `seq_len`.

        Parameters
        ----------
        sequence: Tensor
            The input sequence to warp and pad. Must be a 1-dimensional tensor.

        Returns
        -------
        Tensor
            Output tensor of size `seq_len` + 1 in its second and last
            dimension and as many rows as needed to accommodate all elements
            in its first dimension.

        Notes
        -----
        Sequences that are folded into more than one row will never have any
        row that has fewer than 2 non-padding entries because these would be
        useless in evaluating next-token prediction. However, empty sequences
        or sequences of length 1 will still be padded to a tensor of sizes 1
        and `seq_len` + 1 in its first and second dimension, respectively.
        Consequently, they should be filtered out beforehand!

        """
        return self._pad(sequence).unfold(0, self.width, self.stride)


fold_train = SequenceFolder(
    config.data.seq_len,
    config.data.stride,
    config.data.jitter
)
fold_test = SequenceFolder(config.data.seq_len)
