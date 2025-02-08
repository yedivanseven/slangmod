data
====
The first step to train any language model, large or small, is to get yourself
some data, the cleaner the better. Because :mod:`slangmod` cannot know which
text you want to train your model on, which language(s) that text will be in,
*etc.*, it can do precious little to help clean that text. Before we get to
what it :doc:`can do </cli/clean>`, we will thus specify the format
:mod:`slangmod` expects the text data to be in and where it expects it to be.


format
------
Taking the HuggingFace `dataset collection <https://huggingface.co/datasets>`_
as an example, :mod:`slangmod` expects text data in the form of
`parquet <https://parquet.apache.org/docs/file-format/>`_ files. When read
with, for example, `pandas <https://pandas.pydata.org/>`_ with the help of
`PyArrow <https://arrow.apache.org/docs/python/index.html>`_, this results in
a table (a `DataFrame <https://pandas.pydata.org/pandas-docs/stable/reference/
api/pandas.DataFrame.html>`_). Among the columns in that table, :mod:`slangmod`
expects one to contain the text data, one document per row. More often than
not, the name of that row is "**text**" but, as we will see
:doc:`later </cli/clean>`, this can be configured.


names
-----
Typically, data will be spread out over several such files. Most will be
used to train the model, while some will be used to monitor the training
progress and, if early stopping is active, to terminate training. In addition,
a final evaluation of the model performance will be done on another held-out
validation data set.

Consequently, :mod:`slangmod` expects parquet files that contain (one of) the
words "**train**", "**test**", or "**validation**" in their file names and it
will use these fields accordingly. While configurable, the default file
extension of parquet files is "**.parquet**".

Many data sets on `HuggingFace <https://huggingface.co/datasets>`_ are already
split into files with that naming scheme but, if you want to use one that is
not, you have to split the data yourself and name the files accordingly.

.. important::
   :mod:`slangmod` relies on the presence of all three, **train**, **test**,
   and **validation** files to function properly.


location
---------
If you plan to use your data as is, then all files, test, train, and validation,
should directly go into a folder named "**corpus**" inside :mod:`slangmod`'s
*working directory* that you configured as ``work_dir`` :doc:`earlier <configuration>`.

If, however, you plan to leverage :mod:`slangmod` to do some data cleaning for
you, then your parquet files can stay in any directory that is **not** the
"corpus" folder inside ``work_dir``.

.. note::
   Even if you don't want to do any actual data cleaning with :mod:`slangmod`,
   you can still use the :doc:`/cli/clean` command to simply copy files from
   some source directory into the **corpus** folder.


eos
---
At inference time, you want your model to eventually stop producing next tokens,
ideally when it has said what it wanted to say. One way to realize this is to
stop producing more text when a special "end-of-sequence" (EOS) token
is predicted. However, the model can only do so if there are EOS tokens in
the training data. Too few too far apart and your model will never shut up.
Too many and your model answers might be more concise than you'd like.
Therefore, one important decision to make is what exactly should be considered
a "**sequence**" by your model.

The upper bound for the length of a sequence is the length of a document,
*i.e.*, the contents of rows in the "text" column of your data files.
:mod:`slangmod` will put an EOS token at the end of each. So, if your documents
are rather short (say, a few sentences), you don't have do to anything.
If however, you use much longer documents, like E-books, then you will have to
either deliberately put markers into your documents that designate an EOS,
or identify already existing patterns in your document that :mod:`slangmod`
can interpret as EOS.
