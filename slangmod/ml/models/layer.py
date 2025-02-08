import warnings
from typing import Self, Any
import torch as pt
import torch.nn as ptn
from swak.pt.types import Device, Dtype, Tensor, Block
from swak.pt.misc import Identity
from ...config import LiteralDevice, Devices
from .attention import SelfAttention

__all__ = ['EncoderLayer']


class EncoderLayer(Block):
    """Encoder layer (i.e., self-attention only) to use in a transformer.

    Parameters
    ----------
    attention: SelfAttention
        A suitably parameterized instance of ``SelfAttention``.
    feed_forward: Block
        PyTorch ``Module`` that

        * has a ``reset_parameters()`` method,
        * has a ``new()`` method to make fresh copies of itself,
        * processes tensors with dimensions (..., `S`, `D`),

        where `S` is the sequence length and `D` is the model dimension
        specified in the `attention`.
    pos_enc: Block, optional
        PyTorch ``Module`` that

        * has a ``reset_parameters()`` method,
        * has a ``new()`` method to make fresh copies of itself,
        * has a ``context`` attribute specifying the maximum sequence length,
        * processes tensors with dimensions (..., `S`, `D`),

        where `S` is the sequence length and `D` is the model dimension
        specified in the `attention`. If given, it will be called on the input
        tensor first thing. Typically, this would be an instance of
        ``Sinusoidal`` or ``Learnable`` positional encodings. Defaults to an
        instance of ``Identity``, which does nothing.
    bias: bool, optional
        Whether to use a bias in the ``LayerNorm`` components.
        Defaults to ``True``.
    dropout: float, optional
        Fraction of dropout to apply after self-attention and feed-forward.
        Defaults to 0.1
    norm_first: bool, optional
        Whether to normalize inputs to attention and feed-forward or the sum
        of respective inputs and outputs. Defaults to ``True``.
    eps: float, optional
        Add this value to the denominator in the ``LayerNorm`` components.
        Defaults to 1e-5.
    device: str or device, optional
        Torch device to first create the encoder layer on. Defaults to "cpu".
    dtype: dtype, optional
        Torch dtype to first create the layer in. Defaults to ``torch.float``.

    See Also
    --------
    SelfAttention
    Sinusoidal
    Learnable

    """

    def __init__(
            self,
            attention: SelfAttention,
            feed_forward: Block,
            pos_enc: Block = Identity(),
            bias: bool = True,
            dropout: float = 0.1,
            norm_first: bool = True,
            eps: float = 1e-5,
            device: Device | Devices | LiteralDevice = 'cpu',
            dtype: Dtype = pt.float,
            **_: Any
    ) -> None:
        super().__init__()
        self.attention = attention.to(device=device, dtype=dtype)
        self.feed_forward = feed_forward.to(device=device, dtype=dtype)
        self.pos_enc = self.__check(pos_enc).to(device=device, dtype=dtype)
        self.bias = bias
        self.dropout = dropout
        self.norm_first = norm_first
        self.eps = eps
        self.norm1 = ptn.LayerNorm(
            attention.mod_dim,
            eps=eps,
            elementwise_affine=True,
            bias=bias,
            device=device,
            dtype=dtype
        )
        self.norm2 = ptn.LayerNorm(
            attention.mod_dim,
            eps=eps,
            elementwise_affine=True,
            bias=bias,
            device=device,
            dtype=dtype
        )
        self.drop1 = ptn.Dropout(dropout)
        self.drop2 = ptn.Dropout(dropout)

    def __check(self, pos_enc: Block) -> Block:
        """Warn if both attention and layer apply positional encodings."""
        if not isinstance(pos_enc, Identity) and self.attention.has_pos_enc:
            msg = ("Attention and layer both apply positional encodings! "
                   "Hope you know what you're doing ...")
            warnings.warn(msg)
        return pos_enc

    @property
    def mod_dim(self) -> int:
        """The model dimension."""
        return self.attention.mod_dim

    @property
    def device(self) -> Device:
        """The device all weights, biases, activations, etc. reside on."""
        return self.attention.device

    @property
    def dtype(self) -> Dtype:
        """The dtype of all weights, biases, activations, and parameters."""
        return self.attention.dtype

    @property
    def has_pos_enc(self) -> bool:
        """Whether positional encodings are applied."""
        return (
            self.attention.has_pos_enc or
            not isinstance(self.pos_enc, Identity)
        )

    @property
    def context(self) -> int:
        """Maximum context length given by the positional encodings."""
        if hasattr(self.pos_enc, 'context'):
            return min(self.pos_enc.context, self.attention.context)
        return self.attention.context

    def forward(
            self,
            src: Tensor,
            mask: Tensor | None = None,
            is_causal: bool = True,
    ) -> Tensor:
        """Forward pass of one encoder layer (i.e., with self.attention only).

        Parameters
        ----------
        src: Tensor
            Input sequence(s) of dimensions (..., `S`, `D`), with sequence
            length `S` and model dimension `D`.
        mask: Tensor, optional
            Attention mask with a shape broadcastable to the shape of the
            attention weights (..., `S`, `S`). Two types of masks are
            supported: A boolean mask where a value of ``True`` indicates that
            the element *should* take part in attention or a float mask of the
            same dtype as `src` that is added to the product of queries and
            keys, before taking the softmax. In the latter case, a value of
            0.0 (resulting in unchanged attention weights) indicates that an
            element *should* take part in the attention and a value of "-inf"
            (resulting in a zero attention weight) that it should *not*.
            Defaults to ``None``.
        is_causal: bool, optional
            If set to ``True``, inputs are masked with a `S` x `S` lower
            triangular matrix and `mask` is ignored. Default to ``True``.

        Returns
        -------
        Tensor
            The output has the same shape as the input.

        Important
        ---------
        In adhering to the convention of the `scaled_dot_product_attention
        <https://pytorch.org/docs/stable/generated/torch.nn.functional.
        scaled_dot_product_attention.html>`_, the meaning of ``True`` and
        ``False`` (attend to and *not* attend to, respectively) in boolean
        attention masks is exactly the **opposite** of what it means in the
        `Transformer <https://pytorch.org/docs/stable/generated/torch.nn.
        Transformer.html#torch.nn.Transformer.forward>`_.
        Therefore, to stay compatible, use float masks!

        """
        positioned = self.pos_enc(src)
        if self.norm_first:
            attended = self.attention(self.norm1(positioned), mask, is_causal)
            normed = self.norm2(src + self.drop1(attended))
            out = normed + self.drop2(self.feed_forward(normed))
        else:
            attended = self.attention(positioned, mask, is_causal)
            normed = self.norm1(src + self.drop1(attended))
            out = self.norm2(normed + self.drop2(self.feed_forward(normed)))
        return out

    def reset_parameters(self) -> None:
        """Reset all internal parameters of the layer."""
        self.attention.reset_parameters()
        self.feed_forward.reset_parameters()
        self.pos_enc.reset_parameters()
        self.norm1.reset_parameters()
        self.norm2.reset_parameters()

    def new(self) -> Self:
        """Return a fresh, new instance with exactly the same parameters."""
        return self.__class__(
            self.attention.new(),
            self.feed_forward.new(),
            self.pos_enc.new(),
            self.bias,
            self.dropout,
            self.norm_first,
            self.eps,
            self.device,
            self.dtype
        )
