# PyCMTensor

> A Python tensor based choice modelling package

Python Tensor based package for Deep neural net assisted Discrete Choice Modelling.

![](https://img.shields.io/badge/pycmtensor-0.6.1-orange)

![](https://img.shields.io/pypi/v/pycmtensor.svg)
[![Documentation Status](https://readthedocs.org/projects/pycmtensor/badge/?version=latest)](https://pycmtensor.readthedocs.io/en/latest/?version=latest)
[![](https://pyup.io/repos/github/mwong009/pycmtensor/shield.svg)](https://pyup.io/repos/github/mwong009/pycmtensor)
![Licence](https://img.shields.io/badge/Licence-MIT-blue)

## Features

* Modern optimization algorithms for stochastic gradient descent for estimating choice models with deep neural nets, including Adam, RMSProp and Adadelta.
* Combines conventional econometric models (e.g. Logit) with deep learning models (e.g. ResNet)
* Relatively easy to prototype new deep learning models
* Similar programming syntax as `BIOGEME`, allows easy changes between `BIOGEME` or `PyCMTensor` estimation methods
* Tensor manipulations using `aesara` enable fast estimation on very large scale models

## Install

To install PyCMTensor, you need [Conda](https://docs.conda.io/en/latest/miniconda.html) (Full Anaconda works fine, but **miniconda** is recommmended for a minimal installation)

Once Conda is installed, install the required dependencies from conda by running this 
command in your terminal:

```console
$ conda install -c conda-forge git cxx-compiler m2w64-toolchain libblas libpython
```

>Note: Mac OSX user should install `Clang` for a fast compiled code.

Then, run this command in your terminal to install PyCMTensor:

```console
$ pip install git+https://github.com/mwong009/pycmtensor.git@develop -U
```

This will download and install the latest development branch of PyCMTensor.

## How to use

PyCMTensor uses syntax very similar to `BIOGEME`. Users of `BIOGEME` should be familiar 
with the syntax.

Start an interactive session (IPython or Jupyter Notebook) and import PyCMTensor:
```Python
import pycmtensor as pycmt
```

Several subpackages are also useful to include:
```Python
from pycmtensor.expressions import Beta
from pycmtensor.models import MNLogit
from pycmtensor.optimizers import Adam
from pycmtensor.results import Results
```

## Credits

This package was generated with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage) project template.