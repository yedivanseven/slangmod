tokenize
========
The next step after preparing your data is to train a tokenizer. You can
chose from three of those provided by the HuggingFace `tokenizers <https://
huggingface.co/docs/tokenizers/index>`_ package:

* `unigram <https://huggingface.co/docs/tokenizers/api/models#tokenizers.models.Unigram>`_
* `bpe <https://huggingface.co/docs/tokenizers/api/models#tokenizers.models.BPE>`_ (byte-pair encoding)
* `wordpiece <https://huggingface.co/docs/tokenizers/api/models#tokenizers.models.WordPiece>`_

The default is `unigram <https://huggingface.co/docs/tokenizers/api/models#tokenizers.models.Unigram>`_,
pre-configured with sane settings. You can, therefore, simply invoke:

.. code-block:: bash

   slangmod tokenize

Should you want to use a different one, just pass it to the command line

.. code-block:: bash

   slangmod tokenize --tokens.algo <ALGO>

or put it into a new ``[tokens]`` section in your config TOML file.

.. code-block:: toml
   :caption: slangmod.toml

   work_dir = "/absolute/path/to/your/working/directory"
   log_level = 10
   progress = true

   [files]
   raw = "/absolute/path/to/data/files"
   suffix = "pqt"
   column = "document"
   min_doc_len = 32
   cleaners = ["quotes", "encoding"]
   encoding = "cp1252"

   [tokens]
   algo = "bpe"

:mod:`slangmod` will train a tokenizer and save it into your ``work_dir``.
The name of the file under which it is saved can, in principle, be set with
the ``--files.tokenizer <FILE NAME>`` flag on the command line and you could
can also add your preference to the ``[files]`` section of your config TOML,
but I *strongly* advise *against* setting it explicitly.

.. important::
   In order for your entire workflow to stay consistent, the default name for
   the tokenizer file contains a hash of the entire ``[tokens]`` section,
   including the ``algo``, all :ref:`cli/tokenize:options`, and all settings
   for :ref:`cli/tokenize:eos`.

.. note::
   Should cou choose to set it anyway, be advised that it does not matter
   whether you specify a file extension or not. :mod:`slangmod` will always
   save it with a "json" extension because it is a JSON file.


options
-------
All further configuration options are likewise set with ``--tokens.<KEY> <VALUE>``
on the command line and/or go into the ``[tokens]`` section of your config TOML
file.

tokens.vocab = 16384
   Maximum vocabulary size for *all* tokenizers. For a monolingual model in a
   simple script and clean corpus with few special symbols, this value might
   work. But it is certainly at the lower end._

tokens.dropout = 0.0
   `Dropout <https://arxiv.org/abs/1910.13267>`_ for the
   `BPE <https://huggingface.co/docs/tokenizers/api/models#tokenizers.models.BPE>`_
   tokenizer.

tokens.min_freq = 0
   Minimum frequency a pair should have in order to be merged. Affects both the
   `BPE trainer <https://huggingface.co/docs/tokenizers/api/trainers#tokenizers.trainers.BpeTrainer>`_
   and the
   `WordPiece trainer <https://huggingface.co/docs/tokenizers/api/trainers#tokenizers.trainers.WordPieceTrainer>`_.

tokens.max_len = 16
   Sets the *max_input_chars_per_word* in the
   `WordPiece <https://huggingface.co/docs/tokenizers/api/models#tokenizers.models.WordPiece>`_
   tokenizer, the *max_token_length* in the
   `BPE trainer <https://huggingface.co/docs/tokenizers/api/trainers#tokenizers.trainers.BpeTrainer>`_,
   and the *max_piece_length* in the
   `Unigram trainer <https://huggingface.co/docs/tokenizers/api/trainers#tokenizers.trainers.UnigramTrainer>`_.

tokens.shrink_factor = 0.75
   `shrink_factor` of the
   `Unigram trainer <https://huggingface.co/docs/tokenizers/api/trainers#tokenizers.trainers.UnigramTrainer>`_.

tokens.n_iter = 2
   *n_sub_iterations* of the
   `Unigram trainer <https://huggingface.co/docs/tokenizers/api/trainers#tokenizers.trainers.UnigramTrainer>`_.


eos
---
As discussed :ref:`earlier <usage/data:eos>` you need to let your model know when
a sequence ends. The only way to do that is to tokenize a specific pattern as
a special ``[EOS]`` token. To indicate which pattern that should be, you need
to set two things:

tokens.eos_regex = "\\n{2,}"
   A regular expression that matches the pattern you want to set as EOS.
   Owing to the :ref:`data <cli/clean:cleaners>` I am working with, I decided
   to got with the end of a paragraph, that is, two or more consecutive newline
   characters.

tokens.eos_string = "\\n\\n"
   This is an example string that must match the regular expression you
   just specified.

Again, both can be set either on the command line or in your config file.

.. code-block:: bash

   slangmod tokenize --tokens.eos_string "\n\n" --tokens.eos_regex "\n{2,}"

.. code-block:: toml
   :caption: slangmod.toml

   work_dir = "/absolute/path/to/your/working/directory"
   log_level = 10
   progress = true

   [files]
   raw = "/absolute/path/to/data/files"
   suffix = "pqt"
   column = "document"
   min_doc_len = 32
   cleaners = ["quotes", "encoding"]
   encoding = "cp1252"
   tokenizer = "my_tokenizer.json"

   [tokens]
   algo = "bpe"
   vocab = 30000
   eos_string = "\n\n"
   eos_regex = "\n{2,}"

.. tip::
   Use the excellent `regex 101 <https://regex101.com/>`_ with some
   sample text from your data to make sure both ``tokens.eos_regex`` and
   ``tokens.eos_string`` are correct.
