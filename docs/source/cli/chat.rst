chat
====
Once you have successfully trained model, it is time to talk to it. To do so,
invoke:

.. code-block:: bash

   slangmod chat --name my-first-experiment

All chat options are set with ``--chat.<KEY> <VALUE>`` on the command line
and/or go into a ``[chat]`` section of your config TOML file:

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
   tokenizer = "tokenizer.json"
   checkpoint = "check.pt"
   weights = "state.pt"
   model = "final.pt"

   [tokens]
   algo = "bpe"
   vocab = 30000
   eos_string = "\n\n"
   eos_regex = "\n{2,}"

   [data]
   ...

   [model]
   ...

   [model.feedforward]
   ...

   [train]
   ...

   [chat]
   ...

Keep in mind, though, that the you just *pre*-trained a model. It is not yet
fined-tuned to make for an engaging conversation partner or to excel at any
specific task. It simply continues writing the text from what you prompted it
with. How exactly it does so can be influences by the following parameters.

generator = "greedy"
   Which algorithm to use to generate text. The default "**greedy**", the most
   obvious one, simply picks the most probably next token, one at a time.
   Other options are:

   * "**top_k**" Randomly draw the the next token from the top-k most likely
     ones. The probability with which any of these top-k tokens will be picked
     is proportional to the probability produced by the model.
   * "**top_p**" Randomly draw the the next token from as many of the most
     likely ones as are needed to reach a total probability of :ref:`p`.
   * "**beam**" Perform a beam search for the most probable sequence of tokens.

max_tokens = 256
   How many tokens the model should at most predict if no end-of-sequence
   is encountered first

temperature = 1.0
   Temperature of the (categorical) probability distribution the next token is
   sampled from with the ``top_k`` and ``top_p`` methods. A higher value
   spreads out the probability among candidate tokens, making model responses
   more "creative", while lower values will make responses tend towards ``greedy``.

k = 0.1
   The fraction or number of most likely tokens to draw from for the ``top_k``
   method. Will be interpreted as a fraction if strictly smaller than 1.0 and
   as a count if larger.

.. _p:

p = 0.8
   The sum of the individual probabilities of top most likely next tokens to
   draw from with the ``top_p`` method.

width = 8
   The width of the ``beam`` search.

boost = 1.0
   The *higher* this value, the *longer* answers produced by a ``beam`` search
   will become. Conversely, the *lower* this value, the *shorter* the model
   response will be. Must be larger than 0.0

style = "space"
   Depending on which data you trained your model one, you might still be able
   to affect the style the conversation.

   * "**space**" The model is asked to directly continue whatever text you
     prompt it with, no separator whatsoever,
   * "**paragraph**" An end-of-sequence token is appended to the user input
     to incentivize the model to answer with an entire paragraph itself.
   * "**quotes**" The user prompt is wrapped into double quotes and terminated
     by a comma to incentivize the model to comment on what the user said.
   * "**dialogue**" User input is wrapped in double quotes and an end-of-sequence
     token is appended, thus incentivizing the model to respond with a quoted
     paragraph itself, in the style on might find in a book.

user = "USR"
   The prefix of the user prompt on the console.

bot = "BOT"
   The prefix of the model responses on the console.

stop = "Stop!"
   The text the user should enter to stop the chat client.

system = ""
   The system prompt. This can either be a string or the path to a file with
   the text of the system prompt in it.
