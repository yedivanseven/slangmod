Welcome to slangmod's documentation!
====================================
*Small language model.*

Ever wondered how large language models (LLMs) like ChatGPT, Claude,
LLama, Deepseek, *etc.*, actually work, like, *really* work? If not, then you
will find plenty of libraries and tools that abstract away all the nitty-gritty
details and provide a convenient, high-level playground. If, however, the
answer is yes, then there is only one way to find out, isn't there?
Make one yourself. From scratch.

Of course, you cannot hope to compete with commercial products that take months
and millions to train, but with a little patience, you can actually get
impressive results on consumer hardware. Not any old laptop, mind you, but a
state-of-the art gaming PC with a dedicated graphics card supported by
`PyTorch <https://pytorch.org/>`_ will do.

Specifically, this pacakge provides everything you need to

- preprocess and clean your text corpus;
- chose and train one of the HuggingFace `tokenizers <https://huggingface.co/docs/tokenizers/index>`_;
- specify a Transformer model including the type of positional encodings and the feedforward block;
- train your model with a choice of optimizers and learning-rate schedulers, employing early-stopping if you like;
- monitor convergence and experiment on hyperparameters;
- explore text-generation algorithms like top-k, top-p or beamsearch;
- and, finally, chat with your model.

To that end, :py:mod:`slangmod` provides a command-line interface (CLI) with
fine-grained configuration options on one hand, but also the raw building
blocks it is made of (API) on the other hand. Leveraging the foundational
functionalities provided by the fiercely functional
`swak <https://github.com/yedivanseven/swak>`_ package, any other workflow
can be quickly code up.



.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Usage

   installation
   configuration


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: CLI Reference

   clean
   tokenize
   encode
   train
   chat


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: API Reference

   io
   etl
   ml



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
