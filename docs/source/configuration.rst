configuration
=============
:mod:`slangmod` can be configured *via* its command-line interface (CLI) and/or
*via* a config file in `TOML <https://toml.io/en/>`_ format. All configuration
options can be set by either or both of these mechanisms. In case of
discrepancy, the CLI takes precedence over the config file.

In order to list all the options that you can possible set, simply type

.. code-block:: bash

   slangmod -h

or

.. code-block:: bash

   slangmod --help

As you can see, there's a lot to configure. Some options you will set once
and rarely, if ever, touch again, while you will more frequently adjust others.
I recommend putting the former into a config file, starting with the very basics.
As you move through the steps to your final model, more and more options will
remain static and can be added.

Specifically, you could set

- ``work_dir`` the path to a working directory, where :mod:`slangmod` will
  save all it needs. While, in principle, this path can be *relative*
  (defaulting, in fact, to the directory where you invoke :mod:`slangmod`),
  I recommend setting an absolute path here.
- ``log_level`` the level at which progress is logged to the console. The
  default is 10 (meaning DEBUG, *i.e.*, everything is logged) and I recommend
  leaving it at that. If, however, you want to have fewer messages on the
  console, you can set it to 20 (INFO). Setting it to 30 (WARNING) or higher
  will suppress all log messages.
- ``progress`` some actions (like training a tokenizer or a model) give visual
  feedback in the form of a progress bar in the console by default, *i.e.*,
  when this is set to ``True``. Set to ``False`` if you don't want that.

With these setting, your initial config file could look something like this:

.. code-block:: toml
   :caption: your-config-file.toml

   work_dir = "/absolute/path/to/your/working/directory"
   log_level = 10
   progress = true

To convince yourself that this works, use the ``dry-run`` action on the command
line by invoking :mod:`slangmod` with

.. code-block:: bash

   slangmod dry-run --toml path/to/your-config-file.toml

This does nothing other than printing the configuration that :mod:`slangmod`
would run with. Look for the options you set in the file. They should now have
the desired values. To convince yourself that the CLI takes precedence over
the config file, run

.. code-block:: bash

   slangmod dry-run --toml path/to/your-config-file.toml --log-level 40

and check that the log-level has indeed changed.

And that's it for now. All other commands/actions and configuration options
(and how to set them) are explained in detail in the CLI reference.

.. note::
   In the examples above, we used "long-format" command-line options. With the
   exception of ``-h`` for ``--help``, this is, in fact, the only format accepted
   by :mod:`slangmod`.
