installation
============

- Create and activate a new virtual environment running at least ``python 3.12``.
- The easiest way of installing :mod:`slangmod` is from the python package index
  `PyPI <https://pypi.org/project/slangmod/>`_, where it is hosted. Simply type

  .. code-block:: bash

     pip install slangmod

  or treat it like any other python package in your dependency management.
- While it is, in principle, possible to run :mod:`slangmod` on the CPU, this is
  only intended for debugging purposes. To get any results in finite time, you
  also need a decent graphics card, and you must have a working installation
  of `PyTorch <https://pytorch.org/>`_ to make good use of it. Because there is
  no way of knowing which version of CUDA (or ROC) you have installed on your
  machine and how you installed it, `PyTorch <https://pytorch.org/>`_ it is not an explicit
  dependency of :mod:`slangmod`. You will have to install it yourself, *e.g.*, following
  `these instructions <https://pytorch.org/get-started/locally/>`_.
  If you are using ``pipenv`` for dependency management, you can also have a look at the
  `Pipfile <https://github.com/yedivanseven/slangmod/blob/main/Pipfile>`_ in the root
  of the :mod:`slangmod` `repository <https://github.com/yedivanseven/slangmod>`_ and
  taylor it to your needs. Personally, I go

  .. code-block:: bash

     pipenv sync --categories=cpu

  for a CPU-only installation of PyTorch and

  .. code-block:: bash

     pipenv sync --categories=cuda

  if I want GPU support.
- Finally, with the virtual environment you just created active, open a console
  and type

  .. code-block:: bash

     slangmod -h

  to check that everything works.
