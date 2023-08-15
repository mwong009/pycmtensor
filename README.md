![PyCMTensor](docs/assets/img/logo.jpg)

[![Licence](https://img.shields.io/badge/Licence-MIT-blue)](about/licence.md)
![](https://img.shields.io/pypi/pyversions/pycmtensor) 
[![PyPI version](https://badge.fury.io/py/pycmtensor.svg)](https://badge.fury.io/py/pycmtensor) 
[![codecov](https://codecov.io/gh/mwong009/pycmtensor/branch/master/graph/badge.svg?token=LFwgggDyjS)](https://codecov.io/gh/mwong009/pycmtensor) 
[![Downloads](https://static.pepy.tech/personalized-badge/pycmtensor?period=month&units=international_system&left_color=grey&right_color=orange&left_text=downloads/month)](https://pepy.tech/project/pycmtensor) 
[![DOI](https://zenodo.org/badge/460802394.svg)](https://zenodo.org/badge/latestdoi/460802394)

A Tensor-based choice modelling estimation package written in Python

# Welcome

PyCMTensor is a tensor-optimized discrete choice model estimation Python library 
package, written with optimization compilers to speed up estimation of large datasets, 
simulating very large mixed logit models or implementing neural network functions into 
utility equations in choice models.

Currently, PyCMTensor can be used to fully specify Multinomial Logit and Mixed Logit models, estimate and generate statistical tests, using optimized tensor operations via Aesara tensor libraries.

## Key features

Main features:

- Interpretable and customizable utility specification syntaxes
- Perform statistical tests, analyze covariances for taste parameters.
- Fast execution of model estimation including of simulation based methods, e.g. Mixed Logit models, using Tensor based libraries 
- Model estimation with 1st order (e.g. Adam, Stochastic Gradient Descent) or 1.5th order methods (e.g. Stochastic BFGS)
- Specifying neural nets with weight and bias parameters inside a utility function (TasteNet)

While other choice modelling estimation software in Python are available, e.g. Biogeme, xlogit, PyLogit, etc., PyCMTensor strives to fully implement deep learning based methods written in a simplified syntax for utility equation specification. Different software programs may occasionally vary in their behaviour and estimation results. 

## Documentation

See documentation at https://mwong009.github.io/pycmtensor/