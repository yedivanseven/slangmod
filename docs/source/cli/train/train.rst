train
-----
Training any neural network efficiently and to full convergence is somewhat
of a black art. :mod:`slangmod` gives you full control over the process. The
following options should go into the ``[train]`` section of your config TOML
file, but they can also be configured on the command line with
``--train.<KEY> <VALUE>``.

batch_size = 64
   Size of mini-batches to process. Adjust so that you fully exploit your
   GPU's memory.

step_freq = 1
   Gradients are accumulated for this many batches before the optimizer takes a
   step. Use in case memory shortage necessitates exceedingly small batch sizes.
   For example, you could set it to 2 for a ``batch_size`` of 16, to 4 for a
   ``batch_size`` of 8, and so on.

clip_grad = 0.8
   When the overall norm of all parameter gradients in the model exceeds this
   value, they are scaled back accordingly. This helps stabilize convergence,
   especially in the beginning of a training run.

label_smoothing = 0.1
   Will be forwarded to the `cross-entropy loss <https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss>`_.

optimizer = "adamw"
   Optimizer to use. The choice is between `adamw <https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html#torch.optim.AdamW>`_
   (the default) and `adafactor <https://pytorch.org/docs/stable/generated/torch.optim.Adafactor.html#torch.optim.Adafactor>`_
   (if memory is an issue).

max_epochs = 16
   Maximum number of times that all training data will be shown to the model.

patience = None
   In contrast to what this seems to imply (you have no patience), the default
   is actually to have *no early stopping* and to run for the full ``max_epochs``.
   If you want early stopping to be active, set this to a positive integer > 1,
   indicating the number of consecutive epochs that the test loss has to
   increase before stopping training and retrieving the best checkpoint up
   until that point.

learning_rate = 0.001
   Determines the step size taken by the optimizer in model-parameter space.

warmup = 8_000
   Number of initial mini-batches during which the learning rate is linearly
   ramped up to the specified value.

scaling = "inverse"
   Functional form of how the learning rate is decayed again after reaching
   the specified value. "**inverse**" scales with one over the number of
   mini-batches, "**exponential**" scales with one over a constant to the
   power of the number of mini-batches, and "**cosine**" scales as a quarter
   wave from 1 down to zero.

power = 0.5
   Specifies the negative exponent of the number of mini-batches for "**inverse**"
   scaling. Must be a positive number from the interval [0.5, 1.0].

gamma = 0.95
   The constant to take take the (negative) power of for "**exponential**"
   scaling. Must be smaller than 1.

cooldown = 100_000
   The number of mini-batches until "**cosine**" scaling reaches a learning
   rate of 0.

.. _cb-freq:

cb_freq = 1
   Every how many mini-batches to log the training loss and the current
   learning rate.
