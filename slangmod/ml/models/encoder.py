import warnings
import torch as pt
import torch.nn as ptn
from swak.pt.types import Device, Dtype, Tensor, Tensors1T, Resettable
from swak.pt.misc import Identity
from .layer import EncoderLayer
from ...config import LiteralDevice, Devices

__all__ = ['Encoder']


class Encoder(Resettable):
    """Flexible transformer encoder for natural language modeling.

    Parameters
    ----------
    vocab: int
        The vocabulary size of the tokenizer, i.e., the highest possible token
        id plus one.
    layer: EncoderLayer
        A suitably parameterized instance of ``EncoderLayer``.
    n_layers: int, optional
        How often the `layer` is repeated in the transformer stack.
        Defaults to 2, but must be at least 1.
    pad_id: int, optional
        The id of the padding token. Defaults to 0.
    pos_enc: Resettable, optional
        PyTorch ``Module`` that

        * has a ``reset_parameters()`` method,
        * has a ``context`` attribute specifying the maximum sequence length,
        * processes tensors with dimensions (..., `S`, `D`),

        where `S` is the sequence length and `D` is the model dimension
        specified in the `layer`. If given, it will be called on the input
        tensor first thing. Typically, this would be an instance of
        ``Sinusoidal`` or ``Learnable`` positional encodings. Defaults to an
        instance of ``Identity``, which does nothing.
    bias: bool, optional
        Whether to apply bias in the final projection from the transformer
        output onto the vocabulary size. Defaults to ``True``.
    dropout: float, optional
        Apply dropout to the sum of token embedding and positional encodings
        with this probability during training. Defaults to 0.1
    scale_grad_by_freq: bool, optional
        Whether to scale the gradients on the token embeddings by the inverse
        frequency of their occurrence.
    device: str or device, optional
        Torch device to first create the transformer on. Defaults to "cpu".
    dtype: dtype, optional
        Torch dtype to first create the transformer encoder stack in.
        Defaults to ``torch.float``.

    Raises
    ------
    TypeError
        If neither the encoder itself nor the `layer` applies any positional
        encodings.

    See Also
    --------
    EncoderLayer
    Sinusoidal
    Learnable

    """

    def __init__(
            self,
            vocab: int,
            layer: EncoderLayer,
            n_layers: int = 2,
            pad_id: int = 0,
            pos_enc: Resettable = Identity(),
            bias: bool = True,
            dropout: float = 0.1,
            scale_grad_by_freq: bool = True,
            device: Device | Devices | LiteralDevice = 'cpu',
            dtype: Dtype = pt.float
    ) -> None:
        super().__init__()
        self.vocab = vocab
        self.n_layers = self.__valid(n_layers)
        self.layers = ptn.ModuleList([
            layer.new().to(device=device, dtype=dtype)
            for _ in range(self.n_layers)
        ])
        self.pad_id = pad_id
        self.pos_enc = self.__check(pos_enc).to(device=device, dtype=dtype)
        self.bias = bias
        self.dropout = dropout
        self.scale_grad_by_freq = scale_grad_by_freq
        self.embed = ptn.Embedding(
            num_embeddings=vocab,
            embedding_dim=self.mod_dim,
            padding_idx=pad_id,
            scale_grad_by_freq=scale_grad_by_freq,
            device=device,
            dtype=dtype
        )
        self.drop = ptn.Dropout(dropout)
        self.norm = ptn.LayerNorm(
            self.mod_dim,
            eps=layer.norm1.eps,
            elementwise_affine=layer.norm1.elementwise_affine,
            bias=layer.bias,
            device=device,
            dtype=dtype
        ) if layer.norm_first else Identity()
        self.finalize = ptn.Linear(
            in_features=self.mod_dim,
            out_features=vocab,
            bias=bias,
            device=device,
            dtype=dtype
        )

    @staticmethod
    def __valid(n_layers: int) -> int:
        """Check that the number of layers is at least one."""
        if n_layers < 1:
            msg = 'The transformer must have at least 1 layer, not {}!'
            raise ValueError(msg.format(n_layers))
        return n_layers

    def __check(self, pos_enc: Resettable) -> Resettable:
        """Check compatibility of encoder and layer positional encodings."""
        if not isinstance(pos_enc, Identity) and self.layers[0].has_pos_enc:
            msg = ("Encoder and layer(s) both apply positional encodings! "
                   "Hope you know what you're doing ...")
            warnings.warn(msg)
        if isinstance(pos_enc, Identity) and not self.layers[0].has_pos_enc:
            msg = ('Either the encoder or the layer(s) must apply positional'
                   ' encodings for a transformer architecture to work!')
            raise TypeError(msg)
        return pos_enc

    @property
    def mod_dim(self) -> int:
        """The model dimension."""
        return self.layers[0].mod_dim

    @property
    def device(self) -> Device:
        """The device all weights, biases, activations, etc. reside on."""
        return self.finalize.weight.device

    @property
    def dtype(self) -> Dtype:
        """The dtype of all weights, biases, activations, and parameters."""
        return self.finalize.weight.dtype

    @property
    def context(self) -> int:
        """Maximum context length permitted by the positional encodings."""
        if hasattr(self.pos_enc, 'context'):
            return min(self.pos_enc.context, self.layers[0].context)
        return self.layers[0].context

    def forward(
            self,
            src: Tensor,
            attn_mask: Tensor | None = None,
            src_mask: Tensor | None = None,
            is_causal: bool = True
    ) -> Tensors1T:
        """Forward pass through the transformer encoder with optional masking.

        Parameters
        ----------
        src: Tensor
            Input sequence(s) of token indices. Must be of dtype int64 (=long).
            Expected dimensions are (..., `S`), with `S` the sequence length.
        attn_mask: Tensor, optional
            Floating-point attention mask with a shape broadcastable to the
            shape of the attention weights (..., `S`, `S`) to be added to the
            product of queries and keys, before taking the softmax. A value of
            0.0 (resulting in unchanged attention weights) indicates that an
            element *should* be attended to and a value of "-inf" (resulting
            in a zero attention weight) that it should *not* be attended to.
            Defaults to ``None``.
        src_mask: Tensor, optional
            Floating-point attention mask with a shape broadcastable to the
            shape of `src` (..., `S`). A value of 0.0 indicates that an
            element *should* be attended to and a value of "-inf" that it
            should *not* be attended to. Defaults to ``None``.
        is_causal: bool, optional
            If set to ``True``, inputs are masked with a causal `S` x `S`
            triangular matrix (as produced by `generate_square_subsequent_mask
            <https://pytorch.org/docs/stable/generated/torch.nn.Transformer.
            html#torch.nn.Transformer.generate_square_subsequent_mask>`_) and
            both `attn_mask` and `src_mask` are ignored.
            Defaults to ``True``.

        Returns
        -------
        Tensor
            Un-normalized logits over the next-token probabilities for each
            position with dimensions (..., `vocab`, `S`), where `S` is again
            the sequence length.

        Note
        ----
        Boolean attention masks are not accepted!

        """
        if is_causal or (attn_mask is None and src_mask is None):
            mask = None
        elif src_mask is None:
            mask = attn_mask
        else:
            # Insert a next-to-last dimension to repeat the src_mask in
            reshaped = src_mask.unsqueeze(-2)
            # Construct the arguments to PyTorch tensors' expand method
            sizes = [-1] * reshaped.dim()
            # src_mask will be repeated sequence-length times in new dimension
            sizes[-2] = src_mask.size(-1)
            # Repeat to form a square mask. Shape is now original +1 dim
            src_mask = reshaped.expand(*sizes)
            # Add repeated and reshaped src_mask to attn_mask if present
            mask = src_mask if attn_mask is None else attn_mask + src_mask

        out = self.drop(self.pos_enc(self.embed(src)))
        for layer in self.layers:
            out = layer(out, mask, is_causal)
        return self.finalize(self.norm(out)).transpose(-1, -2).contiguous(),

    def reset_parameters(self) -> None:
        """Reset all learnable parameters in all components of the model."""
        for layer in self.layers:
            layer.reset_parameters()
        self.embed.reset_parameters()
        self.pos_enc.reset_parameters()
        self.finalize.reset_parameters()
