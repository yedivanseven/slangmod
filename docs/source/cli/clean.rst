clean
=====
In order to invoke the ``clean`` command of :mod:`slangmod`, you need to specify
the location of your raw :doc:`data files </usage/data>`. You might have seen
this command-line option when you tried out the CLI for the first time during
basic :doc:`/usage/configuration`:

.. code-block:: json

   "files": {
       "raw": "/directory/where/you/invoked/slangmod",
       "suffix": "parquet",
       "column": "text",
       "min_doc_len": 1,
       "cleaners": [],
       "encoding": "cp1252"
   }

The ``raw`` field, defaulting to whichever directory you invoke :mod:`slangmod`
in, should point towards the folder where your data files are located. As you
can see, however, this field is *nested* inside of the ``files`` struct. So
how do you set this from the command line? Easy. To access nested config
fields just append their names to the top-level field with a dot like so:

.. code-block:: bash

   slangmod clean --files.raw relative/or/absolute/path/to/data/files

Because the location of this directory is probably not going to change that
frequently, it might be a good idea to put it into your config file, again
preferring absolute paths over relative ones.

.. code-block:: toml
   :caption: slangmod.toml

   work_dir = "/absolute/path/to/your/working/directory"
   log_level = 10
   progress = true

   [files]
   raw = "/absolute/path/to/data/files"

Invoking :mod:`slangmod` as described above will do three things:

* It will copy all files with the extension ".parquet" that contain either
  "train", or "test", or "validation" in their names from the ``raw`` folder
  into the "corpus" subdirectory inside your ``work_dir``. It will not descend
  into any subfolders of ``raw``.
* In doing so, it will filter out documents that are shorter than ``min_doc_len``
  characters. Its value defaults to 1 to drop empty documents.
* It will rename your data files with a hash of what is inside them to avoid
  duplicates.

.. warning::
   Every time you invoke ``slangmod clean`` the "corpus" folder inside your
   ``work_dir`` will be completely emptied and re-filled from scratch.
   To *add* more data files instead, you must *resume* cleaning like so:

   .. code-block:: bash

      slangmod resume clean


options
-------
What you can also see is that this is where you can specify the ``suffix``
you use for your parquet files (defaults to "parquet") and the ``column`` in
your data table that contains the actual text (defaults to "text"). To set
these explicitly on the command line, you would go:

.. code-block:: bash

   slangmod clean --files.suffix pqt --files.column document --files.min_doc_len 32

Because again, these options are not going to change very often, you might as well
put them into your config file.

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


.. note::
   It does not matter whether you specify the ``suffix`` with or without a
   leading dot. :mod:`slangmod` will act reasonably.


cleaners
--------
For the data that I have been playing with, english E-books from
`Project Gutenberg <https://www.gutenberg.org/>`_ (provided as
`gutenberg-en-v1-clean <https://huggingface.co/datasets/BEE-spoke-data/gutenberg-en-v1-clean/tree/main/data>`_
by `BEEspoke Data <https://huggingface.co/BEE-spoke-data>`_) and english
Wikipedia articles (a subset of `Wiki-40B <https://aclanthology.org/2020.lrec-1.297.pdf>`_
provided by `google <https://huggingface.co/google>`_ as
`wiki40b <https://huggingface.co/datasets/google/wiki40b/tree/main/en>`_),
I have implemented some actual data cleaning steps. If you plan on using
the same or similar data, then maybe they are useful to you as well.

1. Both, Gutenberg E-books and Wikipedia articles contain "weird" quotes to
   indicate minutes and seconds (*e.g.*, when giving a location with latitude
   and longitude). In addition, Gutenberg E-books sometimes use typographical
   single- and double quotes. I chose to simply replace all of these with
   normal 'single' and "double" quotes, respectively. I you want to do that too,
   invoke the ``quotes`` *cleaner* on the command line like so:

   .. code-block:: bash

      slangmod clean --files.cleaners '["quotes"]'

   If you want to put that into your config file, extend it like so:

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
      cleaners = ["quotes"]

2. I decided that I will use the end of a paragraph, that is, two or
   more consecutive newline characters (``"\n\n"``) as my :ref:`usage/data:eos`
   pattern. Gutenberg E-books are already formatted that way. To also format
   the **wiki40b** articles (and only those!) that way, you can invoke the
   ``wiki40b`` *cleaner* like so:

   .. code-block:: bash

      slangmod clean --files.cleaners '["quotes", "wiki40b"]'

   If you want to put that into your config file too, extend it like so:

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
      cleaners = ["quotes", "wiki40b"]

3. If, like me, you want to start with training a mono-lingual model, then
   having characters from a script in your corpus that is not the main script
   of your primary language unnecessarily blows up your vocabulary size. To
   avoid this, there is a *cleaner* that replaces all characters that cannot be
   encoded with a specified ``encoding`` (defaults to "cp1252") with a
   whitespace. If you want that, you can invoke this cleaner on the command
   line like so:

   .. code-block:: bash

      slangmod clean --files.encoding cp1252 --files.cleaners '["quotes", "wiki40b", "encoding"]'

   If you want to put that into your config file as well, extend it like so:

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
      cleaners = ["quotes", "wiki40b", "encoding"]
      encoding = "cp1252"

.. note::
   Obviously you can pick any combination and order of these *cleaners*.

.. warning::
   The *cleaners* you specify on the command line are **not** *added* to those
   in your config file (or *vice versa*). Rather, the command line overwrites
   the entire list in your config file.

.. important::
   Always double check the data that ends up in your "**corpus**" folder
   and make sure that it adheres to the expected format.
