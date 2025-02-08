encode
======
Now that you have trained a tokenizer, we will use it to encode your text data,
that is, to "translate" each document into a sequence of integers. There are
no configuration options for doing this.

.. code-block:: bash

   slangmod encode

After this step, you should have a subdirectory "**encodings**" in your
``work_dir``, appended by

* the same hash as the tokenizer file (if you didn't explicitly set it) or
* the actual name of the tokenizer file if you did.

This highlights the purpose of that hash. Had you changed any
:ref:`cli/tokenize:options`, :mod:`slangmod` would have complained that it
cannot find a tokenizer file. That way you can track which encoded documents
have run through which tokenizer and you can have different versions.

The reason ``encode`` is a separate step is that it can take a while
(depending on how much data you have) and you don't want to wait around
every time you start a new model-training run. Courtesy to
`swak <https://github.com/yedivanseven/swak>`_ you could also run both, the
:doc:`tokenize` and the ``encode`` step, in one go.

.. code-block:: bash

   slangmod tokenize encode
