![GitHub Pages](https://github.com/yedivanseven/slangmod/actions/workflows/publish-documentation.yml/badge.svg)
![PyPI](https://github.com/yedivanseven/slangmod/actions/workflows/publish-package.yml/badge.svg)

# slangmod
_**Small language model.**_

Ever wondered how large language models (LLMs) like ChatGPT, Claude,
LLama, Deepseek, _etc._, actually work, like, _really_ work? If not, then you
will find plenty of libraries and tools that abstract away all the nitty-gritty
details and provide a convenient, high-level playground. If, however, the
answer is yes, then there is only one way to find out, isn't there?
Make one yourself. From scratch.

Of course, you cannot hope to compete with commercial products that take months
and millions to train, but with a little bit of patience, you can actually get
impressive results on consumer hardware. Not any old laptop, mind you, but a
state-of-the art gaming PC with a dedicated graphics card supported by
[PyTorch](https://pytorch.org/) will do.

Specifically, this pacakge provides everything you need to
- preprocess and clean your text corpus;
- chose and train one of the HuggingFace
  [tokenizers](https://huggingface.co/docs/tokenizers/index);
- specify a Transformer model including the type of positional encodings
  and the feedforward block;
- train your model with a choice of optimizers and learning-rate schedulers,
  employing early-stopping if you like;
- monitor convergence and experiment on hyperparameters;
- explore highly configurable text-generation algorithms like top-k, top-p or beam-search;
- chat with your model.

To that end, `slangmod` provides a command-line interface (CLI) with
fine-grained configuration options on one hand, but also the raw building
blocks it is made of on the other hand. Leveraging the foundational
functionalities provided by the fiercely functional
[swak](https://github.com/yedivanseven/swak) package, any other workflow
can be quickly code up. 


## Installation
* Create a new virtual environment running at least `python 3.12`.
* The easiest way of installing `slangmod` is from the python package index
[PyPI](https://pypi.org/project/slangmod/), where it is hosted. Simply type
  ```shell
  pip install swak
  ```
  or treat it like any other python package in your dependency management.
* While it is, in principle, possible to run `slangmod` on the CPU, this is
  only intended for debugging purposes. To get any results in finite time, you 
  also need a decent graphics card, and you must have a working installation
  of [PyTorch](https://pytorch.org/) to make good use of it. Because there is
  no way of knowing which version of CUDA (or ROC) you have installed on your
  machine and how you installed it, [PyTorch](https://pytorch.org/) is not an
  explicit dependency of `slangmod`. You will have to install it yourself,
  _e.g._, following [these instructions](https://pytorch.org/get-started/locally/).
  If you are using `pipenv` for dependency management, you can also have a
  look at the [Pipfile](https://github.com/yedivanseven/slangmod/blob/main/Pipfile)
  in the root of the `slangmod` [repository](https://github.com/yedivanseven/slangmod)
  and taylor it to your needs. Personally, I go
  ```shell
  pipenv sync --categories=cpu
  ```
  for a CPU-only installation of PyTorch and
  ```shell
  pipenv sync --categories=cuda
  ```
  if I want GPU support.


## Documentation
The documentation for both the CLI and the API of `slangmod` is hosted
on [GitHub Pages](https://yedivanseven.github.io/slangmod/).


## Usage
Use this repository as a template when using the [swak](https://github.com/yedivanseven/swak) package.

First, search for the following tags and replace them as needed:
- `<YEAR>` the year for the copyright notice
- `<AUTHOR>` your name
- `<USER>` your GitHub user name
- `<PACKAGE>` the name of your package, which should match the name of the repository
Then, rename the python package `package` also to <PACKAGE>.

Next, look for and replace other occurrences of
- `package`
- `your@email.com`

Finally, take a good look at the files in the `.github/` folder and taylor them to your needs.
