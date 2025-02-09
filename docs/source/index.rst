Welcome to slangmod's documentation!
====================================
*Small language model.*

Ever wondered how large language models (LLMs) like ChatGPT, Claude,
LLama, Deepseek, *etc.*, actually work, like, *really* work? I did. And I
figured there is only one way to find out: Make one yourself. From scratch.

Of course, I wasn't expecting to beat the big players at their own game,
but I wanted to know what you can do on consumer hardware (meaning a
state-of-the art gaming PC with a single graphics card supported by
`PyTorch <https://pytorch.org/>`_). So, naturally, it was going to be a *small*
language model. These hardware limitations are reflected in software
design choices. Specifically, :mod:`slangmod` does *not* employ any type of
parallelization that would keep multiple GPUs busy at the same time, and *all*
training data are loaded into CPU RAM at once, to be drip-fed to the model
on the GPU from there (1 billion tokens take up about 7.5 GB worth of 64-bit
integer numbers).

Having said that, :mod:`slangmod` provides everything you need to

- preprocess and clean your text corpus;
- chose and train one of the HuggingFace `tokenizers <https://huggingface.co/docs/tokenizers/index>`_;
- specify a Transformer model including the type of positional encodings and the feedforward block;
- train your model with a choice of optimizers and learning-rate schedulers, employing early-stopping if you like;
- monitor convergence and experiment on hyperparameters;
- explore text-generation algorithms like top-k, top-p or beamsearch;
- and, finally, chat with your model.

To do all these things, :mod:`slangmod` provides a command-line interface (CLI)
with fine-grained configuration options on one hand, and the raw building
blocks it is made of on the other hand. Leveraging the foundational
functionalities provided by the `swak <https://github.com/yedivanseven/swak>`_
package, any other workflow can thus be quickly coded up.



.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Usage

   usage/installation
   usage/configuration
   usage/data
   usage/strategy


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: CLI Reference

   cli/clean
   cli/tokenize
   cli/encode
   cli/train
   cli/chat


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: API Reference

   api/io
   api/etl
   api/ml



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
