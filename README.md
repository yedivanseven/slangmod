![GitHub Pages](https://github.com/yedivanseven/slangmod/actions/workflows/publish-documentation.yml/badge.svg)
![PyPI](https://github.com/yedivanseven/slangmod/actions/workflows/publish-package.yml/badge.svg)

# slangmod
_**small language model**_

Ever wondered how large language models (LLMs) like ChatGPT, Claude,
LLama, Deepseek, _etc._, actually work, like, _really_ work? I did. And I
figured there is only one way to find out: Make one yourself. From scratch.

Of course, I wasn't expecting to beat the big players at their own game,
but I wanted to know what you can do on consumer hardware (meaning a
state-of-the art gaming PC with a single graphics card supported by
[PyTorch](https://pytorch.org/)). So, naturally, it was going to be a *small*
language model. These hardware limitations are reflected in software
design choices. Specifically, `slangmod` does *not* employ any type of
parallelization that would keep multiple GPUs busy at the same time, and *all*
training data are loaded into CPU RAM at once, to be drip-fed to the model
on the GPU from there (1 billion tokens take up about 7.5 GB worth of 64-bit
integer numbers).

Having said that, `slangmod` provides everything you need to
- preprocess and clean your text corpus;
- chose and train one of the HuggingFace
  [tokenizers](https://huggingface.co/docs/tokenizers/index);
- specify a Transformer model including the type of positional encodings
  and the feedforward block;
- train your model with a choice of optimizers and learning-rate schedulers,
  employing early-stopping if you like;
- monitor convergence and experiment on hyperparameters;
- explore text-generation algorithms like top-k, top-p or beamsearch;
- and, finally, chat with your model.

To do all these things, `slangmod` provides a command-line interface (CLI)
with fine-grained configuration options on one hand, and the raw building
blocks it is made of on the other hand. Leveraging the foundational
functionalities provided by the [swak](https://github.com/yedivanseven/swak) package, any other workflow
can thus be quickly coded up.


## Installation
### Python package
* Create a new virtual environment running at least `python 3.12`.
* The easiest way of installing `slangmod` is from the python package index
[PyPI](https://pypi.org/project/slangmod/), where it is hosted. Simply type
  ```shell
  pip install slangmod
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
  look at the [Pipfile](https://github.com/yedivanseven/slangmod/blob/main/Pipfile) in the root of the `slangmod` [repository](https://github.com/yedivanseven/slangmod)
  and taylor it to your needs. Personally, I go
  ```shell
  pipenv sync --categories=cpu
  ```
  for a CPU-only installation of PyTorch (for debugging only) and
  ```shell
  pipenv sync --categories=cuda
  ```
  if I want GPU support.
* Finally, with the virtual environment you just created active, open a console
  and type
  ```shell
  slagnmod -h
  ```
  to check that everything works.


### Docker image
A docker image with GPU-enabled [PyTorch](https://pytorch.org/) and all other
dependencies inside is available on the [Docker Hub](https://hub.docker.com/r/yedivanseven/slangmod).
```shell
docker pull yedivanseven/slangmod
```
To use it, you must have a host machine that
* has an NVIDIA GPU,
* has the drivers for it installed, and
* exposes it via the [container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/).

Change into a *working directory*, i.e., one where `slangmod` will read its
config file *slangmod.toml* from and where it will save outputs to, and mount
this directory to the path `/workdir` inside the container when you run it.
```shell
docker run --rm -v ./:/workdir yedivanseven/slangmod
```
This will invoke `slangmod -h`.

In the event that you still want to clean your raw text with the help of
`slangmod`, you will also have to mount the folder with those dirty files
when your start a docker container.
```shell
docker run --rm -v ./:/workdir -v /path/to/raw/docs:/raw yedivanseven/slangmod clean ...
```

For all other command-line options and to find out about this config TOML file,
refer to the ...


## Documentation
The documentation for both the CLI and the API of `slangmod` is hosted
on [GitHub Pages](https://yedivanseven.github.io/slangmod/).
