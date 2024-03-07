# PyCMTensor: An advanced discrete choice modelling package

<p align="center"><img width="460" height="300" src="docs/assets/img/logo.jpg"></p>

[![Conda Forge](https://img.shields.io/conda/vn/conda-forge/pycmtensor?logo=condaforge)](https://anaconda.org/conda-forge/pycmtensor)
![Python Version](https://img.shields.io/badge/dynamic/toml?url=https%3A%2F%2Fraw.githubusercontent.com%2Fmwong009%2Fpycmtensor%2Fmaster%2Fpyproject.toml&query=tool.poetry.dependencies.python&label=Python)
![Conda-Forge|downloads](https://img.shields.io/conda/d/conda-forge/pycmtensor)
![Codecov](https://img.shields.io/codecov/c/github/mwong009/pycmtensor)
[![Licence](https://img.shields.io/badge/Licence-MIT-blue)](about/licence.md)
[![DOI](https://zenodo.org/badge/460802394.svg)](https://zenodo.org/badge/latestdoi/460802394)

## Overview

[PyCMTensor](https://github.com/mwong009/pycmtensor) is an advanced open-source Python library which uses tensor computation to define, operate and estimate neural-embedded discrete choice models. This library provides the tools to fully specify most discrete choice models and then generates comprehensible statistical results for econometric analysis.
[Aesara](https://github.com/aesara-devs/aesara) is used as the backend compiler library. It is based on a computational graph model, making it powerful for use in combining deep neural networks with discrete choice models.

## Getting started

### Installation

**PyCMTensor** is available on Conda-forge:

```bash
conda install -c conda-forge pycmtensor
```

### Documentation

Full documentation is available [here](https://mwong009.github.io/pycmtensor/). 

### Features

PyCMTensor includes:

- A pure-Python implementation for specifying neural networks for various integrations into most discrete choice models.
- Intuitive [Biogeme](https://biogeme.epfl.ch/)-like syntaxes for defining model parameters and structural equations.
- Automatically generate model statistics, covariance matrices, and standard errors.
- A random generator available for simulation-based estimation (e.g. Mixed Logit).
- Implementation for various deep learning optimization methods to estimate complex choice models.

PyCMTensor sets itself apart from other Python based choice modelling estimation software, such as Biogeme, xlogit, or PyLogit, by providing a fully customizable and extensible implementation for choice model utility specification using deep learning methods for estimation.


### Quick start

Examples:
- Swissmetro dataset
  - Multinomial logit ([Example/]())