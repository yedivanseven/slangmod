data
====
Before we can train a model, there are still some decisions to be made
regarding how to feed the data to the model. These settings go into the
``[data]`` section of your config TOML file. Alternatively, you can set options
on the command line with ``--data.<KEY> <VALUE>``.

data.seq_len = 512
   :mod:`slangmod` trains with batches of sequences that are all of the exact
   same length. To that end, they are padded to multiples of this value.

data.overlap = 0.25
   To help the model learn longer, continuous context, consecutive sequences
   can be made to overlap to a certain extent such that the beginning of the
   next sequences is the end of the last. Sequences from different documents
   never mix. This overlap is interpreted as a fraction ``seq_len`` if strictly
   smaller than 1 and as the number of tokens if equal or larger than 1.

data.jitter = 32
   To avoid the model relying on certain patterns at certain positions,
   sequences are randomly shifted by at most this many tokens from one
   training epoch to the next, provided ``shuffle`` is ``False``.

data.shuffle = True
   If ``True`` (the default), files are read in random order, documents are
   are read in random order from each file, the ordering of batches is jumbled
   randomly every epoch, and ``jitter`` is active.

data.device
   Will be set to "**cuda**" if ``torch.cuda.is_available()`` and to "**cpu**"
   if it is not, but can be overridden manually to either fo the two.

data.precision = "bfloat16"
  Defaults to `PyTorch data type <https://pytorch.org/docs/stable/tensors.html>`_
  "**bfloat16**". In case you have too much GPU memory, you can also choose the
  only other permissible option "**float32**".
