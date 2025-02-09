model
=====
Now that the data is ready, on to the main course, the model. :mod:`slangmod`
trains a Transformer with causal self attention, of which you can customize
almost every aspect. The corresponding settings go into a ``[model]``  section
in your config TOML file. Alternatively, you can set them on the command line
with ``--model.<KEY> <VALUE>``.

model.dim = 512
   Model (embedding) dimension.

model.scale_grad_by_freq = True
   If given, scale gradients of the token embeddings by the inverse of their
   frequency the mini-batch.

model.positions = "vanilla"
   Where to use which positional encodings. The default "**vanilla**" uses
   sinusoidal positional encodings added to the raw token embeddings once,
   before they enter the first self-attention block, just like in the original
   `Transformer <https://arxiv.org/html/1706.03762v7>`_ paper. As also
   mentioned there, you can try "**learnable**" positional encodings at this
   location in the model. Further options are "**rotary**", effectively
   changing the model into a `RoFormer <https://arxiv.org/abs/2104.09864>`_,
   and "**sinusoidal**" which adds sinusoidal positional encodings to the
   input of every transformer layer, not just the first.

model.context = 4096
   Positional encodings are never computed on-the-fly but are pre-computed and
   cached up to that length. This is, therefore the maximum number of tokens
   that the model can handle.

   .. warning::

      If you chose "**learnable**" positional encodings, then the context
      must *not* be longer than ``data.seq_len`` because we can only learn
      positional encodings up to the length of the sequences that the model
      sees during training.

model.n_heads = 8
   The number of attention heads to use.

model.n_layers = 8
   The number of Transformer layers to stack.

model.dropout = 0.1
   Amount of `dropout <https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html#dropout>`_
   to apply at various places throughout the model.

model.bias = True
   Whether to add bias to the linear projections all across the model.

model.norm_cls = "layer"
   Which type of norm to use between transformer (sub-)layers. The other choice
   besides the default of "layer" is "rms".

model.norm_first = True
   Whether to normalize the *input* before each Transformer layer or the sum
   of *output* and residual stream after each layer.

model.compile = True
   Whether to compile the model for faster training.


feedforward
-----------
For the feed-forward part of the model, you chose the network architecture,
the involved non-linearities, as well as its size. These settings go into a
nested ``[model.feedforward]`` subsection under ``[model]`` in your config
TOML file. Alternatively, you can set them on the commandline with
``--model.feedforward.<KEY> <VALUE>``.

model.feedforward.flavor = "vanilla"
   As in the in the original `Transformer <https://arxiv.org/html/1706.03762v7>`_
   paper, the input is project up to a size wider than the ``model.dim``, passed
   through a non-linearity, and projected back down to ``model.dim`` with the
   default setting "**vanilla**". Other options are "**glu**", which projects
   the input up and uses one half to gate the other (in a
   `Gated Linear Unit <https://arxiv.org/abs/2002.05202>`_) *before* projecting
   back down to ``model.dim`` and "**grn**", which projects up first, applies
   a non-linearity, and then applies gating *after* projecting back down
   to ``model.dim``, reminiscent of a
   `Gated Residual Network <https://arxiv.org/html/2405.16177v1>`_.

model.feedforward.activation = "gelu"
   Non-linearity to use for the "**vanilla**" and "**grn**" up-projections.
   Other options are "**elu**", "**relu**", "**swish**", and "**mish**.

model.feedforward.gate = "gelu"
   Non-linearity to use for the gating in "**glu**" and "**grn**". Other
   options are "**sigmoid**", "**elu**", "**relu**", "**swish**", "**mish**",
   but also "**none**", which result in a *bilinear* unit. This option is
   disregarded when using the "**vanilla**" feed-forward layer.

model.feedforward.factor = 4
   The width of the hidden layer in the feed-forward network expressed as a
   multiple of ``model.dim``.
