![PyCMTensor](docs/assets/img/logo.jpg)

[![Licence](https://img.shields.io/badge/Licence-MIT-blue)](about/licence.md)
![](https://img.shields.io/pypi/pyversions/pycmtensor) 
[![PyPI version](https://badge.fury.io/py/pycmtensor.svg)](https://badge.fury.io/py/pycmtensor) 
[![codecov](https://codecov.io/gh/mwong009/pycmtensor/branch/master/graph/badge.svg?token=LFwgggDyjS)](https://codecov.io/gh/mwong009/pycmtensor) 
[![Downloads](https://static.pepy.tech/personalized-badge/pycmtensor?period=month&units=international_system&left_color=grey&right_color=orange&left_text=downloads/month)](https://pepy.tech/project/pycmtensor) 
[![DOI](https://zenodo.org/badge/460802394.svg)](https://zenodo.org/badge/latestdoi/460802394)

# PyCMTensor: Advanced Tensor-based Choice Modelling in Python

Welcome to PyCMTensor, a Python library for tensor-based discrete choice modelling estimation. This package is designed with a focus on hybrid neural networks and Logit models, including Mixed Logit models. PyCMTensor models leverage computational graphs and are estimated using generalized backpropagation algorithms.

With PyCMTensor, you can fully specify Multinomial Logit and Mixed Logit models, perform model estimation using computational graphs, and generate statistical test results for econometric analysis.

## Key Features

- **Interpretable and customizable utility specification syntaxes**: Easily define your models with an intuitive syntax.
- **Neural network specification**: Specify neural networks with weight and bias parameters inside utility functions (e.g., TasteNet).
- **Comprehensive analysis tools**: Perform specification testing, analyze covariances, and compute standard errors for taste parameters.
- **Fast model estimation**: Quickly estimate models, including simulation-based methods like Mixed Logit models, using a computational graph approach.
- **Flexible optimization methods**: Tune the model estimation with 1st order methods (e.g., Adam, Stochastic Gradient Descent) or 1.5th order methods (e.g., Stochastic BFGS).

While other choice modelling estimation software in Python are available, such as Biogeme, xlogit, and PyLogit, PyCMTensor sets itself apart by fully implementing deep learning-based methods with a simplified syntax for utility equation specification.

## Documentation

For more information on how to use PyCMTensor, please refer to our [documentation](link-to-documentation).